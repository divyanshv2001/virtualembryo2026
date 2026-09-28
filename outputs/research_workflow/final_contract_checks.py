"""Loop-2 arithmetic counterexamples; no operational access enforcement."""
import json
import math
from pathlib import Path
here = Path(__file__).resolve().parent
checks = {}
stages = {'A':(.6,[(10,9,1),(20,2,5),(5,0,None)]), 'B':(.4,[(8,2,3),(12,6,8)])}
p = {s:sum(k/n for n,k,_ in rows)/len(rows) for s,(_,rows) in stages.items()}
pi = sum(a*p[s] for s,(a,_) in stages.items())
mass = 0
identities = []
for s,(a,rows) in stages.items():
    denom = sum(k/n for n,k,_ in rows)
    for n,k,_ in rows:
        if k:
            rowmass = pi*(a*p[s]/pi)*((k/n)/denom)/k
            identities.append(abs(rowmass-a/(len(rows)*n))<1e-12)
            mass += rowmass*k
checks['unequal_capture_and_absent_strata_row_mass_identity'] = all(identities)
checks['mixture_stratum_mass_identity'] = abs(mass-pi)<1e-12
checks['zero_stratum_excluded_from_conditional_draw'] = stages['A'][1][2][1] == 0
mu_a = sum((k/n)*mu for n,k,mu in stages['A'][1] if k)/sum(k/n for n,k,_ in stages['A'][1])
checks['conditional_mean_uses_proportion_weights'] = abs(mu_a-1.4)<1e-12
q, lam, ma, mb = .6,.5,0,10
targetmean = q*ma+(1-q)*mb
reca, recb = ma+lam*(targetmean-ma), mb+lam*(targetmean-mb)
checks['equal_proportion_mean_unchanged'] = abs(q*reca+(1-q)*recb-targetmean)<1e-12
checks['between_endpoint_variance_quartered'] = abs(q*(1-q)*(reca-recb)**2-.25*q*(1-q)*(ma-mb)**2)<1e-12
ma,mb,qa = 1.8,4,.75
mixmean, tmean = qa*ma+(1-qa)*mb, .6*ma+.4*mb
checks['unequal_proportion_mean_changes'] = abs((1-lam)*mixmean+lam*tmean-2.515)<1e-12
def nearest(points):
    return [min(abs(a-b) for j,b in enumerate(points) if i!=j) for i,a in enumerate(points)]
checks['pairing_does_not_validate_pooled_frame'] = nearest([0,1,0,1]) != nearest([0,1,10,11])
x = [math.log1p(1),math.log1p(2)]
checks['nonclosed_panel_fails_roundtrip_gate'] = abs(sum(math.expm1(v) for v in x)-10)>1e-12
reserved = '__AUDIT_MISSING__'
checks['reserved_token_collision_requires_refusal'] = reserved in ['typeA',reserved]
checks['annotation_provenance_mismatch_requires_refusal'] = {'method':'v1'} != {'method':'v2'}
result = {'checks':checks, 'passed':all(checks.values()), 'scope':'Deterministic arithmetic/counterexample checks. No assay validity, scorer conformance, biological result, durable access refusal or clean-context isolation tested.', 'counterexample':{'conditional_mean_A':mu_a,'equal_proportion_mean':targetmean,'unequal_proportion_baseline_mean':mixmean,'unequal_proportion_recentered_mean':2.515}}
(here/'final_contract_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
raise SystemExit(not result['passed'])
