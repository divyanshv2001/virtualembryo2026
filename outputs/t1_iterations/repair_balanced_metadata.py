"""Correct legacy cohort label field without altering sampled expression or original evidence."""
import json
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import append_event

def main():
    folder=HERE/'private/balanced_proxy_prepared_01';path=folder/'report.json'
    report=json.loads(path.read_text());plan=json.loads((folder/'plan.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(folder/name)!=report[key]:raise ValueError('Prepared data changed')
    if 'original_report_sha256' in report:return
    original=folder/'report.executed.json'
    if original.exists():raise ValueError('Do not overwrite original evidence')
    original.write_bytes(path.read_bytes())
    report['original_report_sha256']=digest(original)
    report['domain_labels']=[label for labels in plan['proxy_groups'].values() for label in labels]
    report['proxy_groups']=plan['proxy_groups'];report['per_group_cap']=plan['per_group_cap']
    report['metadata_correction']='Legacy cardiac-only domain_labels display field replaced by actual frozen proxy_groups labels. Sample rows, genes, expression and original plan/events unchanged.'
    report['metadata_correction_code_sha256']=digest(HERE/'repair_balanced_metadata.py')
    path.write_text(json.dumps(report,indent=2))
    append_event(folder/'events.jsonl','cohort_report_display_metadata_corrected',original_report_sha256=report['original_report_sha256'],corrected_report_sha256=digest(path))

if __name__=='__main__':main()
