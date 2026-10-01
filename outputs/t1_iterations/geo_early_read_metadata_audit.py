"""Read only GEO/SRA metadata for four already selected strict early pilots."""
import json
import re
import urllib.request

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def fetch(url, path):
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read(2_000_001)
    if len(content) > 2_000_000:
        raise ValueError('Metadata response exceeds bound')
    path.write_bytes(content)
    decoded = content.decode()
    if 'recaptcha/challengepage' in decoded.lower():
        raise ValueError('Metadata retrieval returned CAPTCHA challenge; no bypass attempted')
    return decoded


def main():
    out = HERE/'private/geo_early_read_metadata_audit_01'
    if out.exists():
        raise ValueError('Preserve prior metadata audit')
    out.mkdir()
    source = HERE/'GSE76118_EARLY_SCHEMA_RESULTS.json'
    samples = json.loads(source.read_text())['outcomes']
    plan = {'created_utc': now(), 'source_sha256': digest(source),
            'code_sha256': digest(HERE/'geo_early_read_metadata_audit.py'),
            'gsm_ids': [s['gsm'] for s in samples], 'max_bytes_per_metadata_response': 2_000_000,
            'scope': 'GEO/SRA HTML metadata only, no sequencing or expression downloads, no training.'}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_early_read_metadata_audit.py').read_bytes())
    events = out/'events.jsonl'
    append_event(events, 'plan_frozen', plan_sha256=digest(out/'plan.json'))
    rows = []
    for sample in samples:
        gsm = sample['gsm']
        row = {'gsm': gsm, 'stage': sample['stage']}
        try:
            geo_url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+gsm
            geo_path = out/(gsm+'.geo.html')
            geo = fetch(geo_url, geo_path)
            ids = set(re.findall(r'SRX\d+', geo))
            if len(ids) != 1:
                raise ValueError('Expected one SRA experiment relation')
            srx = next(iter(ids))
            sra_url = 'https://www.ncbi.nlm.nih.gov/sra?term='+srx
            sra_path = out/(gsm+'.sra.html')
            html = fetch(sra_url, sra_path)
            plain = re.sub(r'<[^>]+>', ' ', html)
            plain = re.sub(r'\s+', ' ', plain)
            # Archived summary counts spots, not sequencing reads; retain units.
            summary = re.search(r'(\d+) ILLUMINA.*?run:\s*([\d,]+) spots', plain)
            if not summary:
                raise ValueError('Archived run/spot summary unavailable')
            runs, spots = int(summary[1]), int(summary[2].replace(',', ''))
            paired = bool(re.search(r'Layout:\s*PAIRED', plain))
            row.update(status='parsed', geo_url=geo_url, sra_url=sra_url, srx=srx,
                geo_sha256=digest(geo_path), sra_sha256=digest(sra_path),
                archived_run_count=runs, archived_spots=spots, paired_layout=paired,
                paired_read_count_estimate=2*spots if paired else None,
                archived_depth_below_one_million_even_counting_pairs=paired and 2*spots < 1_000_000,
                original_analysis_cell_inclusion=None,
                interpretation='Spot count is archive metadata, not an exact reconstruction of original sequencing/QC. Paired read estimate assumes two reads per spot.')
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:250])
        rows.append(row)
        append_event(events, 'pilot_metadata_audited', **row)
    report = {'status': 'completed', 'updated_utc': now(), 'outcomes': rows,
              'plan_sha256': digest(out/'plan.json'), 'models_fit': 0, 'full_panel_scores': 0,
              'training_ready': False, 'reward_delta': 0,
              'next_gate': 'Freeze prospective QC using explicit archive spot/paired-read units and assigned protein fractions; resolve stable-ID panel mapping and historical cohort selection before expanded early-cell acquisition.'}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (HERE/'GSE76118_EARLY_READ_METADATA_RESULTS.json').write_text(json.dumps(dict(report, report_sha256=digest(out/'report.json')), indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
