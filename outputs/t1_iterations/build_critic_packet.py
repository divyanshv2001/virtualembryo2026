"""Minimal per-metric handoff; no data/history or source files sent to critics."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
METRICS=('de_score','de_direction','mmd_u','variogram')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--metric',required=True,choices=METRICS);parser.add_argument('--out',required=True);args=parser.parse_args()
    report_path=Path(args.report).resolve();out=Path(args.out).resolve()
    if not report_path.is_relative_to(HERE) or not out.is_relative_to(HERE):raise ValueError('Use project evidence/output paths only')
    raw=report_path.read_bytes();report=json.loads(raw);rows=[]
    for fold in report.get('folds',[]):
        for result in fold.get('results',[]):
            rows.append({'cutoff':fold.get('cutoff'),'target':fold.get('target'),'held_capture':fold.get('held_capture'),'candidate':result.get('candidate'),
                'headline':result.get('local_score'),'raw':result.get('raw_metrics',{}).get(args.metric),
                'skill':result.get('skills',{}).get(args.metric),'calibration_valid':result.get('calibration_valid')})
    if not rows:
        for result in report.get('summary',[]):
            rows.append({'candidate':result.get('candidate'),'headline':result.get('scores'),
                'raw':[r.get(args.metric) for r in result.get('raw_metrics',[])],
                'skills':[r.get(args.metric) for r in result.get('skills',[])],
                'calibrations_valid':result.get('all_calibrations_valid')})
    packet={'report_sha256':hashlib.sha256(raw).hexdigest(),'metric':args.metric,'plan_sha256':report.get('plan_sha256'),
        'raw_direction':'lower_better' if args.metric in ('mmd_u','variogram') else 'higher_better',
        'skill_direction':'higher_better','outcomes':rows,'metric_available':any(any(v is not None for v in row.get('raw',[])) if isinstance(row.get('raw'),list) else row.get('raw') is not None for row in rows),
        'passing_candidates':report.get('passing_candidates'),'specialist_role_version':2,'instructions':'In<=60words state Problem / Proposed solution / Validation: evidence vs uncertainty, concrete past-only remedy, matched controls, success/failure criteria. Missing metrics unavailable; never invent outcomes/credentials. No history rereads. Readiness unchanged.'}
    if report.get('audit_context'):
        packet['audit_context']=report['audit_context']
    if report_path.name.startswith('PRODUCTION_COVERAGE_AUDIT'):
        packet['audit_context']={k:report.get(k) for k in ('status','error','time_policy','donor_provenance_passed','gene_order_passed','strata','source_anchor_comparison','unchanged_protected_genes_verified','mapped_mass_max_relative_error','mean_unmapped_anchor_abundance_fraction','reported_head_support_counts_only','scope')}
    if report_path.name.startswith('ONE_DAY_DECODER_DIAGNOSTIC'):
        packet['diagnostic_context']={k:report.get(k) for k in ('status','error','generation','scores_are_historical_replays','new_scores','reward','scope')}
        selected={'de_score':['mean_absolute_expression_error','mean_absolute_detection_error'],
                  'de_direction':['gene_mean_direction_agreement_epsilon1e6','predicted_detection_change_sum','actual_detection_change_sum'],
                  'mmd_u':['mean_absolute_expression_error','mean_absolute_variance_error'],
                  'variogram':['mean_absolute_variance_error','mean_absolute_detection_error']}[args.metric]
        reduced=[]
        diagnostics=report.get('diagnostics',[])
        for candidate in sorted({r['candidate'] for r in diagnostics}):
            group=[r for r in diagnostics if r['candidate']==candidate]
            reduced.append({'candidate':candidate,'historical_panel_count':len(group),'strata_panel_means':{
                label:{key:sum(row['strata'][label][key] for row in group)/len(group) for key in selected}
                for label in group[0]['strata']}})
        packet['diagnostic_context']['diagnostic_proxies_not_benchmark_metrics']=reduced
    encoded=json.dumps(packet,separators=(',',':'))+'\n'
    if len(encoded.encode())>6000:raise ValueError('Split large batch by predeclared experiment; do not pass oversized packet')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(encoded)
    print(json.dumps({'metric':args.metric,'rows':len(rows),'bytes':len(encoded.encode()),'out':str(out)}))

if __name__=='__main__':main()
