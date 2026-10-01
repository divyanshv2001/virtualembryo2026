"""Stream transformation diagnostics; negative preclip values are never forecasts."""
import numpy as np
from robust_population import covariance_change


def trace(donors,pred,mapped,labels,by_group,cov_features,panel):
    n=len(donors);size=len(mapped);raw_global=np.zeros(size);clip_global=np.zeros(size);final_global=np.zeros(size)
    lineages=[];component_arrays={};clipped_mass=np.zeros(n)
    raw_cov=np.asarray(donors[:,cov_features],float).copy();inverse={int(c):j for j,c in enumerate(mapped)}
    for group,shift in by_group.items():
        rows=np.flatnonzero(labels==group);raw=np.zeros(size);clipped=np.zeros(size);final=np.zeros(size)
        strata={name:dict(entries=0,raw_sum=0.,clipping_increment_sum=0.,repair_increment_sum=0.) for name in ['donor_zero','donor_positive','preclip_negative','preclip_nonnegative']}
        for start in range(0,size,256):
            cols=mapped[start:start+256];sl=slice(start,start+len(cols));a=np.asarray(donors[np.ix_(rows,cols)],float)
            before=a+shift[sl];after=np.maximum(before,0).astype(np.float32);b=np.asarray(after,float);c=np.asarray(pred[np.ix_(rows,cols)],float)
            raw[sl]=(before-a).mean(0);clipped[sl]=(b-a).mean(0);final[sl]=(c-a).mean(0)
            clipped_mass[rows]+=np.expm1(b).sum(1)
            for name,mask in [('donor_zero',a==0),('donor_positive',a>0),('preclip_negative',before<0),('preclip_nonnegative',before>=0)]:
                d=strata[name];d['entries']+=int(mask.sum());d['raw_sum']+=float((before-a)[mask].sum());d['clipping_increment_sum']+=float((b-before)[mask].sum());d['repair_increment_sum']+=float((c-b)[mask].sum())
        raw_global+=len(rows)*raw/n;clip_global+=len(rows)*clipped/n;final_global+=len(rows)*final/n
        expected=shift!=0
        reversals=lambda actual:int(((np.sign(actual)==-np.sign(shift))&expected&(np.abs(actual)>1e-8)).sum())
        lineages.append({'lineage':group,'cells':len(rows),'preclip_max_mean_change':float(np.max(np.abs(raw))),
             'postclip_max_mean_change':float(np.max(np.abs(clipped))),'postrepair_max_mean_change':float(np.max(np.abs(final))),
             'postclip_sign_reversals_above_1e8':reversals(clipped),'postrepair_sign_reversals_above_1e8':reversals(final),
             'requested_mean_shift_energy':float(raw@raw),'postclip_mean_shift_energy':float(clipped@clipped),'postrepair_mean_shift_energy':float(final@final),
             'boundary_strata':strata})
        component_arrays[group]=(raw,clipped,final)
        for j,c in enumerate(cov_features):
            if int(c) in inverse:raw_cov[rows,j]+=shift[inverse[int(c)]]
    # Unsupported rows are unchanged and contribute their original count mass.
    unsupported=~np.isin(labels,list(by_group))
    for start in range(0,size,256):clipped_mass[unsupported]+=np.expm1(np.asarray(donors[np.ix_(np.flatnonzero(unsupported),mapped[start:start+256])],float)).sum(1)
    clipped_cov=np.maximum(raw_cov,0).astype(np.float32)
    final_mean=(pred[:,mapped].mean(0,dtype=float)-donors[:,mapped].mean(0,dtype=float))
    np.testing.assert_allclose(final_global,final_mean,atol=1e-10)
    clip_increment=clip_global-raw_global;repair_increment=final_global-clip_global
    np.testing.assert_allclose(raw_global+clip_increment+repair_increment,final_global,atol=1e-12)
    top=np.argsort(-np.abs(final_global),kind='stable')[:10]
    genes=[{'gene':panel[int(mapped[j])],'mapped_position':int(j),'preclip_shift':float(raw_global[j]),'clipping_increment':float(clip_increment[j]),'repair_increment':float(repair_increment[j]),'final_shift':float(final_global[j]),
            'lineage_components':{g:{'requested':float(v[0][j]),'postclip':float(v[1][j]),'postrepair':float(v[2][j])} for g,v in component_arrays.items()}} for j in top]
    maxima=[float(np.max(np.abs(v))) for v in [raw_global,clip_global,final_global]]
    return {'preclip_invalid_negative_entries':sum(v['boundary_strata']['preclip_negative']['entries'] for v in lineages),
         'preclip_is_valid_forecast':False,'stages':['requested_preclip_invalid_diagnostic','after_clip','after_count_mass_repair'],
         'global_max_mean_changes':maxima,'global_mean_guard_passes':[v<=.5 for v in maxima],
         'encoder_feature_covariance_changes':[covariance_change(donors[:,cov_features],raw_cov),covariance_change(donors[:,cov_features],clipped_cov),covariance_change(donors[:,cov_features],pred[:,cov_features])],
         'lineages':lineages,'top_guard_genes':genes,'mean_component_additivity_verified':True}


def controls():
    a=np.array([[0.,1.],[1.,0.]],np.float32);labels=np.array(['fit','fit']);cols=np.arange(2)
    result=trace(a,a,cols,labels,{'fit':np.zeros(2)},cols,['a','b'])
    assert result['global_max_mean_changes']==[0.,0.,0.]
    b=np.maximum(a+np.array([-.5,.5]),0).astype(np.float32)
    result=trace(a,b,cols,labels,{'fit':np.array([-.5,.5])},cols,['a','b'])
    assert result['preclip_invalid_negative_entries']==1 and result['mean_component_additivity_verified']


if __name__=='__main__':
    controls();print('Copy identity, invalid-boundary count and mean-component additivity controls passed')
