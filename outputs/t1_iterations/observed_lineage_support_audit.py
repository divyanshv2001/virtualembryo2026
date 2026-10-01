"""Observed E8.5 tissue-proxy source/challenge support audit; no future reads."""
import json
from collections import Counter
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage

CHALLENGE_MAP={'Foregut':'endoderm','pSHF':'cardiopharyngeal','aSHF':'cardiopharyngeal','JCF':'cardiopharyngeal',
 'OFT/RV-CM':'cardiomyocyte','IFT-CM':'cardiomyocyte','AVC-CM':'cardiomyocyte','RV-CM':'cardiomyocyte','LV-CM':'cardiomyocyte','SV-CM':'cardiomyocyte',
 'Endothelium':'endothelial','Blood':'blood','NCC':'neural_crest','Surface Ectoderm':'ectoderm_other','Neural Tube':'ectoderm_other',
 'Paraxial Mesoderm':'mesenchymal','EXEM':'mesenchymal','Pericardium':'mesenchymal'}


def q(x):return {str(v):float(np.quantile(x,v)) for v in [.05,.5,.95]}
def rho(a,b):return float(spearmanr(a,b).statistic)


def main():
    root=HERE.parents[1];source=HERE/'private/associated_prepared_01';out=HERE/'private/observed_lineage_support_01'
    if out.exists():raise ValueError('Preserve original audit')
    anchor=root/'data/E8.5_RNA.h5ad';panel_path=root/'outputs/t1_run/T1__val.genes.txt';encoder_path=HERE/'private/cnf_hurdle_temporal_01/cutoff_8.25/encoder.npz'
    plan={'created_utc':now(),'code_sha256':digest(HERE/'observed_lineage_support_audit.py'),'source_report_sha256':digest(source/'report.json'),
        'challenge_sha256':digest(anchor),'encoder_sha256':digest(encoder_path),'panel_sha256':digest(panel_path),'encoder_training_cutoff':8.25,
        'source_stage':8.5,'challenge_stage':8.5,'challenge_label_map':CHALLENGE_MAP,'source_map_source_sha256':digest(HERE/'lineage_residual_screen.py'),
        'minimum_source_group_cells':30,'minimum_cross_capture_reference_cells':10,'source_radius_quantile':.95,'batch_cells':128,
        'metric':'Euclidean8D fixed past-trained latent coordinates. Source calibration nearest same-group cell from a DIFFERENT capture. Challenge nearest same-group source cell, compared with source-only95%radius. Also unconstrained source NN proxy-label concordance. No prediction/model/scorer fitting.',
        'scope':'Descriptive coarse tissue-proxy mapping, not validated identities or independent embryo evidence. E8.5 allowed observations only; no E9.5/future reads, scores, reward or causal transfer proof.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(out/'executed_source.py').write_bytes((HERE/'observed_lineage_support_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    manifest=json.loads((source/'report.json').read_text())
    for filename,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/filename)!=manifest[key]:raise ValueError('Source changed: '+filename)
    panel=panel_path.read_text().splitlines();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1};mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in mapped])
    with np.load(encoder_path) as enc:
        features=enc['features'].copy();feature_atlas=enc['atlas_features'].copy();center=enc['center'].copy();scale=enc['scale'].copy();pc=enc['pca_center'].copy();basis=enc['basis'].copy()
    if not np.array_equal(feature_atlas,np.array([lookup[panel[i]] for i in features])):raise ValueError('Encoder gene mapping mismatch')
    if len(basis)!=8:raise ValueError('Expected frozen8D encoder')
    metadata=pd.read_csv(source/'selected_metadata.csv');sr=metadata.index[metadata.numeric_stage==8.5].to_numpy();sl=metadata.loc[sr,'celltype_extended_atlas'].map(lineage).to_numpy();captures=metadata.loc[sr,'sample'].to_numpy()
    x=np.load(source/'expression.npy',mmap_mode='r');a=ad.read_h5ad(anchor,backed='r')
    def collect(matrix,rows,genes,feature_genes,labels):
        groups=sorted(set(labels));aggregate={g:{'count':0,'sum':np.zeros(len(genes)),'positive':np.zeros(len(genes)),'detection':[],'mass':[]} for g in groups};latent=np.empty((len(rows),8))
        for start in range(0,len(rows),128):
            subset=rows[start:start+128];raw=matrix[subset,:][:,genes];raw=raw.toarray() if sparse.issparse(raw) else np.asarray(raw);raw=raw.astype(np.float32,copy=False)
            f=matrix[subset,:][:,feature_genes];f=f.toarray() if sparse.issparse(f) else np.asarray(f)
            latent[start:start+len(subset)]=(((f-center)/scale)-pc)@basis.T
            if not np.isfinite(raw).all() or np.any(raw<0):raise ValueError('Invalid observed expression')
            chunklabels=labels[start:start+len(subset)]
            for g in groups:
                mask=chunklabels==g
                if not mask.any():continue
                block=raw[mask];v=aggregate[g];v['count']+=len(block);v['sum']+=block.sum(0,dtype=np.float64);v['positive']+=(block>0).sum(0);v['detection'].extend((block>0).sum(1).tolist());v['mass'].extend(np.expm1(block).sum(1,dtype=np.float64).tolist())
        return latent,aggregate
    try:
        if not a.var_names.is_unique or a.var_names.tolist()!=panel:raise ValueError('Challenge panel mismatch')
        original=a.obs.celltype.fillna('unannotated').astype(str);cl=original.map(CHALLENGE_MAP).fillna('unmapped_label').to_numpy()
        sz,sagg=collect(x,sr,atlas,feature_atlas,sl);cz,cagg=collect(a.X,np.arange(a.n_obs),mapped,features,cl)
    finally:a.file.close()
    global_nn=NearestNeighbors(n_neighbors=1,n_jobs=2).fit(sz);global_distance,index=global_nn.kneighbors(cz);nearest_labels=sl[index[:,0]]
    results=[]
    for g in sorted(set(sl)|set(cl)):
        smask=sl==g;cmask=cl==g;ns=int(smask.sum());nc=int(cmask.sum());record={'tissue_proxy':g,'source_cells':ns,'challenge_cells':nc,'source_fraction':ns/len(sl),'challenge_fraction':nc/len(cl),
            'source_label_names':sorted(metadata.loc[sr[smask],'celltype_extended_atlas'].astype(str).unique().tolist()),'challenge_label_names':sorted(original[cmask].unique().tolist())}
        if min(ns,nc)<30:record['support_status']='insufficient_cells';results.append(record);continue
        distances=[];covered=0
        for capture in np.unique(captures[smask]):
            query=smask&(captures==capture);reference=smask&(captures!=capture)
            if reference.sum()<10:continue
            d=NearestNeighbors(n_neighbors=1,n_jobs=2).fit(sz[reference]).kneighbors(sz[query])[0][:,0];distances.extend(d.tolist());covered+=int(query.sum())
        record['source_cross_capture_covered_cells']=covered
        if not distances:record['support_status']='cross_capture_reference_unavailable';results.append(record);continue
        radius=float(np.quantile(distances,.95));d=NearestNeighbors(n_neighbors=1,n_jobs=2).fit(sz[smask]).kneighbors(cz[cmask])[0][:,0]
        sa=sagg[g];ca=cagg[g];record.update(support_status='diagnosed',source_cross_capture_nn_distance_quantiles=q(distances),source_only95_radius=radius,
            challenge_same_proxy_nn_distance_quantiles=q(d),challenge_fraction_beyond_source95_radius=float((d>radius).mean()),
            challenge_unconstrained_nn_proxy_label_agreement=float((nearest_labels[cmask]==g).mean()),
            source_detection_quantiles=q(sa['detection']),challenge_detection_quantiles=q(ca['detection']),
            detection_median_ratio=float(np.median(ca['detection'])/max(np.median(sa['detection']),1e-12)),
            mapped_implied_count_median_ratio=float(np.median(ca['mass'])/max(np.median(sa['mass']),1e-12)),
            gene_prevalence_rho=rho(sa['positive']/ns,ca['positive']/nc),gene_mean_log1p_rho=rho(sa['sum']/ns,ca['sum']/nc))
        results.append(record);append_event(events,'tissue_support_diagnosed',**record)
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'scope':plan['scope'],'source_cells':len(sl),'challenge_cells':len(cl),
        'mapped_genes':len(mapped),'unmapped_challenge_labels':sorted(original[cl=='unmapped_label'].unique().tolist()),'results':results,'new_scores':0,'reward_delta':0,'independent_embryo_validation':False}
    np.savez_compressed(out/'observed_latent_coordinates.npz',source=sz,challenge=cz)
    report['latent_coordinates_sha256']=digest(out/'observed_latent_coordinates.npz')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');report['report_sha256']=digest(out/'report.json')
    (HERE/'OBSERVED_LINEAGE_SUPPORT_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'completed':True,'report_sha256':report['report_sha256']}))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
