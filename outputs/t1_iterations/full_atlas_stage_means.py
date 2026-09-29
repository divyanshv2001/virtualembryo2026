"""Stream full source-atlas genes into stage means without a dense cell matrix.

Only source stages through E8.5 are used; all outputs remain on D:.
"""
from collections import Counter
import json
import numpy as np
import pandas as pd

from train_extended_atlas import HERE
from prepare_extended_atlas import decode_record
from cohort_groups import GROUPS
from run_t1 import digest
from iterate import now, append_event


STAGES=(7.5,7.75,8.,8.25,8.5)


def main():
    source=HERE/'private/extended_atlas'
    out=HERE/'private/full_atlas_stage_means_01'
    if out.exists():raise ValueError('Preserve previous streaming run')
    root=HERE.parents[1]
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    meta=pd.read_csv(source/'meta.tsv',sep='\t',usecols=['stage','celltype_extended_atlas'],low_memory=False)
    stage=pd.to_numeric(meta.stage.astype(str).str.replace('E','',regex=False),errors='coerce').to_numpy(float)
    rows=np.flatnonzero(np.isin(stage,STAGES));stage_codes=np.searchsorted(STAGES,stage[rows])
    cardiac=meta.celltype_extended_atlas.isin(GROUPS['cardiac']).to_numpy()[rows]
    count_whole=np.bincount(stage_codes,minlength=len(STAGES))
    count_cardiac=np.bincount(stage_codes[cardiac],minlength=len(STAGES))
    if (count_whole<100).any() or (count_cardiac<50).any():raise ValueError('Insufficient stage support')
    index=json.loads((source/'exprMatrix.json').read_text())
    genes=sorted(((key,int(value[0]),int(value[1])) for key,value in index.items() if '|' in key),key=lambda x:x[1])
    symbol_counts=Counter(key.split('|',1)[1] for key,_,_ in genes)
    panel_index={s:i for i,s in enumerate(panel)}
    mapped=[(key,offset,length,panel_index[key.split('|',1)[1]]) for key,offset,length in genes
            if symbol_counts[key.split('|',1)[1]]==1 and key.split('|',1)[1] in panel_index]
    plan={'created_utc':now(),'stages':STAGES,'source_rows':len(rows),
          'stage_whole_counts':count_whole.tolist(),'stage_cardiac_counts':count_cardiac.tolist(),
          'genes_all_for_library':len(genes),'genes_mapped_for_means':len(mapped),
          'normalization':'log1p(10000 * raw count / complete indexed-gene library count) per source cell',
          'first_pass':'Sum all indexed genes for eligible source-cell library sizes; no future stages.',
          'second_pass':'Mapped official genes only; aggregate full-source stage means for whole and published cardiac proxy cells.',
          'source_sha256':{f:digest(HERE/f) for f in ['full_atlas_stage_means.py','prepare_extended_atlas.py','cohort_groups.py']},
          'input_sha256':{f:digest(source/f) for f in ['meta.tsv','exprMatrix.json','exprMatrix.bin']},
          'panel_sha256':digest(root/'outputs/t1_run/T1__val.genes.txt'),
          'limitations':'Cardiac proxy labels from source atlas are not one-to-one challenge ontology matches. This is sufficient-statistic preparation, not target prediction or validation.',
          'storage':'D-only small library and pseudobulk arrays; no dense full-atlas matrix.',
          'submissions_allowed':0,'jev_requests_allowed':0}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    state_path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(state_path.read_text())
    state.setdefault('full_atlas_stage_means_job',{}).update(
        status='running',run=out.name,plan_sha256=digest(out/'plan.json'),
        source_rows=len(rows),genes_mapped=len(mapped),storage='D-only small sufficient statistics')
    state['active_jobs']=[out.name];state['active_run_path']=f'private/{out.name}'
    state['local_process_running']=True
    state['next_experiment']='Stream all permitted E7.5-E8.5 source cells into stage means; then run frozen historical trend transfer checks before any challenge forecast.'
    state_path.write_text(json.dumps(state,indent=2))
    library=np.zeros(len(rows),dtype=np.float64)
    with (source/'exprMatrix.bin').open('rb') as handle:
        for g,(_,offset,length) in enumerate(genes):
            handle.seek(offset)
            _,vector=decode_record(handle.read(length),len(meta))
            library+=vector[rows]
            if (g+1)%4000==0:append_event(events,'library_pass_progress',genes=g+1,total=len(genes))
    if (library<=0).any() or not np.isfinite(library).all():raise ValueError('Invalid complete library sizes')
    np.save(out/'source_library_sizes.npy',library,allow_pickle=False)
    append_event(events,'library_pass_completed',min_library=float(library.min()),median_library=float(np.median(library)))
    whole=np.zeros((len(STAGES),len(panel)),dtype=np.float32)
    card=np.zeros_like(whole);mapped_mask=np.zeros(len(panel),dtype=bool)
    with (source/'exprMatrix.bin').open('rb') as handle:
        for g,(key,offset,length,panel_col) in enumerate(mapped):
            handle.seek(offset)
            _,vector=decode_record(handle.read(length),len(meta))
            values=np.log1p(vector[rows].astype(np.float64)*(10000/library))
            whole[:,panel_col]=(np.bincount(stage_codes,weights=values,minlength=len(STAGES))/count_whole).astype(np.float32)
            card[:,panel_col]=(np.bincount(stage_codes[cardiac],weights=values[cardiac],minlength=len(STAGES))/count_cardiac).astype(np.float32)
            mapped_mask[panel_col]=True
            if (g+1)%4000==0:append_event(events,'mean_pass_progress',genes=g+1,total=len(mapped))
    np.savez_compressed(out/'stage_means.npz',whole=whole,cardiac=card,mapped=mapped_mask,
                        stages=np.array(STAGES),whole_counts=count_whole,cardiac_counts=count_cardiac)
    report={'status':'completed','plan_sha256':digest(out/'plan.json'),
            'stage_means_sha256':digest(out/'stage_means.npz'),
            'library_sha256':digest(out/'source_library_sizes.npy'),
            'stage_whole_counts':count_whole.tolist(),'stage_cardiac_counts':count_cardiac.tolist(),
            'genes_mapped':int(mapped_mask.sum()),'max_training_stage':8.5,
            'full_source_rows':len(rows),'full_matrix_retained':False,'future_expression_used':False}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    append_event(events,'full_source_means_completed',**report)
    state=json.loads(state_path.read_text())
    state['full_atlas_stage_means_job'].update(status='completed',report_sha256=digest(out/'report.json'),
        stage_means_sha256=report['stage_means_sha256'])
    state['active_jobs']=[];state['local_process_running']=False
    state_path.write_text(json.dumps(state,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        folder=HERE/'private/full_atlas_stage_means_01'
        if folder.exists():
            (folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc)},indent=2))
            append_event(folder/'events.jsonl','execution_failed',type=type(exc).__name__,error=str(exc))
        raise
