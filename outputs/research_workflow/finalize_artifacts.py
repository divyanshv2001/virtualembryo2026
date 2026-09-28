"""Explicit final amendments; original source and secret contents remain untouched."""
import hashlib
import json
from pathlib import Path
here = Path(__file__).resolve().parent
root = here.parents[1]
def write(name,obj):
    (here/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')

design=(here/'EXPERIMENT_DESIGN_REV1.md').read_text(encoding='utf-8')
design=design.replace('revision 1','final revision 2')
design=design.replace('composition-only endpoint mixture','endpoint mixture')
design=design.replace('combined conditional-state shift','shrunk endpoint recentering')
design=design.replace('MMD(comp,target_e)−MMD(combined,target_e)','MMD(endpoint_mixture,target_e)−MMD(shrunk_endpoint_recentering,target_e)')
design=design.replace('corrections/loop1_genmodel.md defines the recipe','corrections/loop2_genmodel.md corrects the donor weights/means from the loop-1 recipe')
design=design.replace('then embryo and whole row. Composition-only equals the convex endpoint joint mixture','then embryo with probability p_esk/Σ_e p_esk and a uniform stratum row. Endpoint-mixture equals the convex equal-embryo endpoint expression mixture')
design=design.replace('Shared strata use μ_tk=', 'Use μ_sk=Σ_e p_esk μ_esk/Σ_e p_esk. Shared strata use μ_tk=')
design=design.replace('Both methods receive the same output map.', 'Both methods receive the same output map. Panel closure, denominator domain, log base, L and zero-row policy must all be documented; a wider denominator stops this map. Annotation method/version, reserved-token collision refusal and allowed missing-label mass must be frozen. Incompatible provenance stops this stratified analysis.')
design=design.replace('Keep donor coordinates; pairing does not align incompatible frames, predict growth or prove neighborhood fidelity.', 'Keep donor coordinates only as uninterpreted placeholders unless a training-derived common frame, physical units and coverage pass separate spatial-readiness gates. Pairing does not validate pooled neighborhoods or predict growth. No target-derived registration is allowed.')
design += '\n## Loop-2 controlling boundaries\n\nSYNTHESIS_2.md controls final interpretation. The mixture already changes conditional expression distributions; recentering tests a frozen transformation, not identified composition/state biology. With equal proportions it can leave mean unchanged and quarter between-endpoint variance before output mapping.\n\nThe access flag/event sequence is a policy illustration only. Durable enforcement and clean-context isolation are unimplemented and untested. Real launch must reject repeat assessment, changed locks and excluded reads under immutable hashes. Expression descriptive/confirmatory and spatial readiness are separate predicates in the final template. Secondary metrics remain descriptive; guardrails are conditional/not applicable.\n'
(here/'EXPERIMENT_DESIGN.md').write_text(design,encoding='utf-8')

report=(here/'REPORT.md').read_text(encoding='utf-8')
report=report.replace('Require a documented zero-row policy, training round-trip check and overflow rejection.', 'Require documented panel closure/denominator domain, log base, L, zero-row policy, training round-trip and overflow rejection. Wider-panel normalization fails this closure route. Reserve a collision-checked missing token, freeze allowed missing-label mass and verify comparable annotation method/version before fitting.')
lines=report.splitlines()
for i,line in enumerate(lines):
    if line.startswith('Loop 2 corrected conditional') or line.startswith('Loop 2 findings and final dispositions'):
        lines[i]='Loop 2 corrected conditional embryo weighting/means, narrowed the contrast to recentering rather than biological decomposition, separated spatial readiness, expanded denominator/annotation gates, withdrew operational access-test claims and reconciled registry/template semantics. Eleven arbitrary arithmetic/counterexample checks passed. Durable access enforcement and clean-context isolation remain unimplemented and untested. Final dispositions are recorded in critics/loop_2.md, critics/issues_2.json and critics/resolution_2.md. SYNTHESIS_2 is the final controlling synthesis. No third critic loop is performed. Corrections establish a more defensible scoped audit/protocol, not empirical truth.'
report='\n'.join(lines)+'\n'
(here/'REPORT.md').write_text(report,encoding='utf-8')

issues=json.loads((here/'critics/issues_2.json').read_text(encoding='utf-8'))
for issue in issues:
    issue['status']='documentary_or_arithmetic_corrected_empirical_operational_risk_open'
    issue['resolution_artifact']='critics/resolution_2.md'
write('critics/issues_2.json',issues)

ledger=json.loads((here/'evidence_ledger.json').read_text(encoding='utf-8'))
extras=[('E073','First-loop arithmetic and policy illustrations pass; they are not access enforcement.','baseline_contract_results.json'),('E074','Final corrected sampling/variance/frame/domain counterexamples pass; biological validity is untested.','final_contract_results.json')]
for eid,claim,source in extras:
    if not any(e['id']==eid for e in ledger):
        ledger.append({'id':eid,'claim':claim,'evidence':source,'source':source,'source_type':'executed arbitrary synthetic arithmetic/counterexample','date':'2026-09-28','confidence':'high for narrow arithmetic scope','supporting_experts':['orchestrator','genmodel'],'contradicting_experts':[],'status':'observed synthetic checks; not biological or operational evidence'})
write('evidence_ledger.json',ledger)
phases=[json.loads((here/f'jev_{p}.json').read_text()) for p in ('inventory','loop1','loop2')]
usage={k:sum(p['usage'][k] for p in phases) for k in ('input_tokens','output_tokens')}
verification=json.loads((here/'verification.json').read_text())
verification['initial_configuration_snapshot']={'api_key_present':False,'live_requests':0}
verification['jev']={'replacement_file_present':True,'live_requests':3,'returned_model':'jev-1.13.0','usage':usage,'scope':'explicitly approved advisory routing only; no scientific evidence'}
verification['final_contract_checks']={'file':'final_contract_results.json','passed':11,'biological_experiments':0,'operational_access_enforcement_tested':False}
write('verification.json',verification)

queue=[]
actions=[('provenance','Obtain clean context and screened provenance before competition design','blocked_pending_clean_context'),('data','Acquire eligible arrays and independent sample/litter/batch metadata','blocked_missing_inputs'),('domain','Verify annotation, panel closure, normalization and coordinate contracts','blocked_missing_inputs'),('scorer','Obtain versioned scorer/hash and exact metrics','blocked_missing_inputs'),('estimator','Implement corrected equal-embryo endpoint sampler and recentering','prospective'),('access','Implement durable locks, access ledger and actual refusal tests','prospective'),('statistics','Replicated eligible pilot; freeze margins/variance/analysis','blocked_missing_inputs'),('primary','Assess frozen single proxy once; preserve negative outcomes','blocked_not_ready'),('engineering','Repair missing-answer gates, gate-first dedupe and measured/idempotent lifecycle','prospective'),('validation','Independent imaging/state/lineage/intervention tests matched to claim','prospective')]
for i,(owner,action,status) in enumerate(actions,1):
    queue.append({'id':f'Q{i:02}','priority':i,'owner':owner,'action':action,'status':status,'executed':False,'scientific_status':'not biological evidence'})
write('research_queue.json',queue)

log=here/'DECISION_LOG.md'
text=log.read_text(encoding='utf-8')
marker='12. **Final workflow completion:**'
if marker not in text:
    text+='\n12. **Final workflow completion:** Exactly two independent critic loops and expert corrections completed. Correct conditional donor weights and recentering interpretation supersede loop-1 recipe; original source remains unchanged. Access/isolation enforcement is future work, not tested.\n13. **Approved Jev budget:** Exactly three live requests completed; aggregate '+str(usage['input_tokens'])+' input and '+str(usage['output_tokens'])+' output tokens. Final routing choice final_risk_review (.99) is advisory. No further request is authorized by this three-request approval.\n14. **Commit requested:** User explicitly requested committing relevant work. Stage scientific audit artifacts, generated proposal/debate records and unchanged original prompt/code needed for reproducibility. Secrets, environments, caches and platform metadata remain ignored. No push was requested.\n'
    log.write_text(text,encoding='utf-8')
manifest={'started':'2026-09-27','finalized':'2026-09-28','workflow':'agenticprompt adopted by user','experts_completed':16,'paired_discussions':8,'critic_loops_completed':2,'critic_revisions_completed':2,'biological_experiments_executed':0,'original_source_modified':False,'live_jev_requests':3,'jev_usage':usage,'scientific_contribution':'source-grounded audit and prospective falsifiable protocol','competition_eligibility':'unresolved; current contexts exposed','access_enforcement':'not implemented or tested','source_inventory':'inventory.json','final_report':'REPORT.md','final_synthesis':'SYNTHESIS_2.md','proposal_count':28,'evidence_entries':len(ledger),'readiness':{'descriptive_expression':False,'confirmatory_expression':False,'spatial':False},'scope_note':'Historical synthesized design files are preserved; SYNTHESIS_2, final design/template/report control interpretation.'}
files=[]
for directory in (here,root/'outputs/orchestration/round_1/proposals'):
    for p in sorted(directory.rglob('*')):
        rel=p.relative_to(root)
        if p.is_file() and not any(part in ('.secrets','.venv','__pycache__') for part in p.parts) and p.name not in ('run_manifest.json','completion_check.json'):
            files.append({'path':rel.as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest['artifact_hashes']=files
manifest['original_source_repository']={'url':'https://github.com/divyansh469/virtualembryo2026','commit':'78caa1b','local_directory':'Neurl IPS 2026','note':'Separate existing Git repository; preserved, not added as an unconfigured gitlink.'}
write('run_manifest.json',manifest)
print(json.dumps({k:manifest[k] for k in ('experts_completed','critic_loops_completed','live_jev_requests','evidence_entries','readiness')}))
