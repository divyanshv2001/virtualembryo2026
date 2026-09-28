from index_scores import outcomes


def test_ledger_keeps_invalid_trials_and_fold_context():
    report={'folds':[{'cutoff':8.,'target':9.,'floor':{'mmd_u':1},'ceiling':{'mmd_u':0},
        'results':[{'candidate':'failed','local_score':None,'skills':None,'calibration_valid':False,
                    'raw_metrics':{'mmd_u':2},'invalid_calibration_metrics':['de_score']},
                   {'candidate':'valid','local_score':51,'skills':{'mmd_u':.51},'calibration_valid':True}]}]}
    rows=list(outcomes(report))
    assert len(rows)==2 and rows[0]['local_score'] is None
    assert rows[0]['cutoff']==8 and rows[0]['target']==9
    assert rows[0]['floor']=={'mmd_u':1}
    assert rows[0]['invalid_calibration_metrics']==['de_score']
    assert rows[1]['raw_metrics'] is None
