"""Refresh small continuation context; append meaningful updates without deleting evidence."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone


def refresh(update=None):
    p=Path(__file__).resolve().parent;s=json.loads((p/'LOCAL_OPTIMIZATION_STATE.json').read_text())
    stamp=datetime.now(timezone.utc).isoformat();export=s.get('latest_progress_export',{})
    lines=['# T1 compact research checkpoint','',f'Updated: {stamp}','',
      f"Official best reported: **{s.get('best_reported_official_score',51.87)}**. Latest export: `{export.get('artifact','none')}`. Local>72 gate remains unmet.",
      'Official metrics: DE44.9, direction57.0, MMD53.5, variogram51.7. Latest screenshot shows S2WISH2 / Codex / AGENT, rank29 on that view; scoring-page rank131/270 is a different view.',
      '',f"Next: {s.get('next_experiment','Review current state')}",'',f"Active jobs: {json.dumps(s.get('active_jobs',[]),separators=(',',':'))}",'',
      '## Latest measured decisions','',
      '- Anchor slope joint both.5:54.9083 local vs original53.5953; official progress51.87. Earlier-fold49.7722/51.2691 vs control49.1752/51.6250: inconsistent temporal gain.',
      '- Kernel population integration:33scores; all8 variants46.50–49.93 below frozen54.91. Rejected bounded integration; family remains open.',
      '- Log1p positive decoder: conditionalridge1; positiveanchoralpha0/.5 crossedabundance/joint, detectionalpha.5 andflow fixed.21scores, no target fitting.',
      '', '## Token and evidence policy','',
      'Read this file first; inspect only active plan, changed policy/state/queue entries and relevant frozen reports. Avoid full-history replay and unchanged large JSON output. Batch independent small reads. Write every meaningful update to RESEARCH_UPDATES.md and refresh this checkpoint.',
      'Keep all raw metrics/skills/calibration/splits/seeds/hashes in SCORE_LEDGER.jsonl and source reports. Summaries never replace genuine trajectories or original private predictions.',
      'Use deterministic rules for routine status. Jev is advisory compact routing, cached by full canonical packet/questions/model. No calls for unchanged state;3000byte cap, one attempt/no retries in this renewed authorization turn. No datasets/source files/secrets sent. Future unattended heartbeat Jev prohibition remains unchanged.',
      '', '## Jev accounting','']
    f=p/'JEV_ROUTE_20260929_01.json'
    if f.exists():
        r=json.loads(f.read_text());lines.append(f"Latest: {r.get('status')}; {r.get('payload_bytes')} payload bytes; actual usage {r.get('usage')}; answers {r.get('answers')}. One live attempt, original3audit calls separate. No measured Codex token savings claim.")
    current=s.get('active_run_path','')
    if current:
        run=Path(current).name
        for key,job in s.items():
            if isinstance(job,dict) and job.get('run')==run:
                lines+=['','## Current experiment','',f"`{run}`: {job.get('status','unknown')}; planned {job.get('evaluations_planned','unknown')}, completed {job.get('evaluations',0)}."]
                report=job.get('results_report')
                if report and (p/report).exists():
                    result=json.loads((p/report).read_text())
                    means=[f"{row['candidate']}={row.get('mean_score')}" for row in result.get('summaries',[])]
                    lines.append('; '.join(means));lines.append(f'Full metrics: [{report}]({report}).')
                break
    if s.get('research_storage'):
        lines+=['','## Storage','',json.dumps(s['research_storage'],separators=(',',':'))]
    lines+=['','## Evidence links','',
       '- [State](LOCAL_OPTIMIZATION_STATE.json), [queue](METRIC_RESEARCH_QUEUE.json), [mandate](LOCAL_OPTIMIZATION_PROMPT.md).',
       '- [Slope results](CNF_ANCHOR_SLOPE_RESULTS.json), [temporal](CNF_ANCHOR_SLOPE_TEMPORAL_RESULTS.json), [kernel](CNF_KERNEL_POPULATION_RESULTS.json).',
       '- [Official feedback](../t1_submissions/anchorslope_progress_20260929_01/official_result.json).',
       '- Full32285gene scorer unchanged;>=64frozenMC mean AND2.5thpercentile>72, metricmeans>=50, matched temporal gains required. Local72≠official. No automatic official submissions or agents. Never invent quota resets.']
    (p/'COMPACT_RESEARCH_CHECKPOINT.md').write_text('\n'.join(lines)+'\n')
    if update:
        log=p/'RESEARCH_UPDATES.md'
        if not log.exists():log.write_text('# T1 research updates\n\nConcise updates; complete metrics remain in linked source reports and SCORE_LEDGER.jsonl.\n')
        with log.open('a') as handle:handle.write(f'\n## {stamp}\n\n{update}\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--update');args=parser.parse_args();refresh(args.update)
