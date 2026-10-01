"""Bounded historical annotation acquisition and four-cell QC audit; no fits."""
import gzip
import argparse
import json
import re
import urllib.request
from collections import Counter

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event

URL = 'https://ftp.ensembl.org/pub/release-79/gtf/mus_musculus/Mus_musculus.GRCm38.79.gtf.gz'
MAX_COMPRESSED = 40_000_000
MAX_EXPANDED = 1_000_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-bounded-acquisition', action='store_true')
    args = parser.parse_args()
    out = HERE/('private/geo_early_biotype_audit_02' if args.reuse_bounded_acquisition else 'private/geo_early_biotype_audit_01')
    if out.exists():
        raise ValueError('Prior audit retained; do not overwrite or duplicate acquisition')
    out.mkdir()
    schema_path = HERE/'GSE76118_EARLY_SCHEMA_RESULTS.json'
    plan = {'created_before_acquisition_utc': now(), 'url': URL,
            'code_sha256': digest(HERE/'geo_early_biotype_audit.py'),
            'schema_sha256': digest(schema_path),
            'max_compressed_bytes': MAX_COMPRESSED, 'max_expanded_bytes': MAX_EXPANDED,
            'reuses_prior_download': args.reuse_bounded_acquisition,
            'scope': 'Gene annotation only, existing four strict CD1 pilots only. No expanded expression acquisition or training.',
            'qc_definition': 'Protein-coding assigned-count fraction with unknown-biotype bounds; not original total-read QC.',
            'training_ready': False}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_early_biotype_audit.py').read_bytes())
    events = out/'events.jsonl'
    append_event(events, 'plan_frozen', plan_sha256=digest(out/'plan.json'))
    try:
        archive = (HERE/'private/geo_early_biotype_audit_01' if args.reuse_bounded_acquisition else out)/'Mus_musculus.GRCm38.79.gtf.gz'
        size = 0
        if args.reuse_bounded_acquisition:
            if digest(archive) != '53bc4bb37067ef52596d05c51ab6b01f1aaa3f27184b1ef938812f3a27522221':
                raise ValueError('Prior annotation checksum mismatch')
            size = archive.stat().st_size
        else:
            with urllib.request.urlopen(URL, timeout=30) as response, archive.open('wb') as target:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_COMPRESSED:
                        raise ValueError('Annotation exceeds compressed bound')
                    target.write(chunk)
        append_event(events, 'annotation_acquired', bytes=size, sha256=digest(archive))
        genes = {}
        expanded = 0
        with gzip.open(archive, 'rt') as stream:
            for line in stream:
                expanded += len(line.encode())
                if expanded > MAX_EXPANDED:
                    raise ValueError('Annotation exceeds expanded bound')
                if line.startswith('#'):
                    continue
                fields = line.rstrip().split('\t')
                if len(fields) != 9:
                    raise ValueError('Invalid GTF columns')
                if fields[2] != 'gene':
                    continue
                attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
                gene = attrs['gene_id'].split('.')[0]
                value = {'symbol': attrs.get('gene_name', ''),
                         'biotype': attrs.get('gene_biotype', attrs.get('gene_type', ''))}
                if gene in genes and genes[gene] != value:
                    raise ValueError('Conflicting gene annotation')
                genes[gene] = value
        if len(genes) < 30_000:
            raise ValueError('Unexpectedly incomplete gene annotation')
        compact = out/'gene_annotation.json'
        compact.write_text(json.dumps(genes, separators=(',', ':'))+'\n')
        panel_path = HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
        panel = panel_path.read_text().splitlines()
        symbols = Counter(r['symbol'] for r in genes.values() if r['symbol'])
        unique = {g:r['symbol'] for g,r in genes.items() if r['symbol'] and symbols[r['symbol']]==1}
        outcomes = []
        schema = json.loads(schema_path.read_text())
        for sample in schema['outcomes']:
            if sample['status'] != 'parsed':
                continue
            path = HERE/'private/geo_early_schema_audit_01'/(sample['gsm']+'.HTSeq.output.txt.gz')
            if digest(path) != sample['file_sha256']:
                raise ValueError('Pilot expression changed')
            counts = {}
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    gene, value = line.rstrip().split('\t')
                    if not gene.startswith('__'):
                        counts[gene] = float(value)
            total = sum(counts.values())
            protein = sum(v for g,v in counts.items() if genes.get(g,{}).get('biotype')=='protein_coding')
            unknown = sum(v for g,v in counts.items() if not genes.get(g,{}).get('biotype'))
            matched = set(unique[g] for g in counts if g in unique) & set(panel)
            outcomes.append({'gsm': sample['gsm'], 'stage': sample['stage'],
                'assigned_count_sum': total, 'protein_coding_assigned_fraction': protein/total,
                'unknown_biotype_assigned_fraction': unknown/total,
                'protein_fraction_bounds': [protein/total, (protein+unknown)/total],
                'protein_fraction_at_least_80_percent': protein/total >= .8,
                'assigned_count_below_one_million': total < 1_000_000,
                'total_sequencing_reads': None, 'original_paper_qc_passed': None,
                'unique_historical_symbol_panel_genes': len(matched),
                'historical_symbol_panel_coverage_fraction': len(matched)/len(panel),
                'missing_annotation_is_zero': False})
        report = {'status': 'completed', 'updated_utc': now(),
            'plan_sha256': digest(out/'plan.json'), 'annotation_url': URL,
            'archive_sha256': digest(archive), 'archive_bytes': size,
            'expanded_bytes_streamed': expanded, 'gene_annotation_sha256': digest(compact),
            'annotated_genes': len(genes), 'biotype_counts': dict(Counter(r['biotype'] for r in genes.values())),
            'panel_sha256': digest(panel_path), 'outcomes': outcomes,
            'models_fit': 0, 'full_panel_scores': 0, 'reward_delta': 0, 'training_ready': False,
            'limitation': 'Assigned gene sums do not establish original total sequencing reads. Historical symbol coverage is incomplete; four selected cells do not establish population transfer.',
            'next_gate': 'Resolve original per-cell read/QC metadata and freeze prospective selection and identifier mapping before expanding the early CD1 cohort.'}
    except Exception as exc:
        report = {'status': 'failed', 'updated_utc': now(), 'error_type': type(exc).__name__,
                  'error': str(exc)[:300], 'plan_sha256': digest(out/'plan.json'),
                  'training_ready': False, 'models_fit': 0, 'full_panel_scores': 0}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    public = dict(report, report_sha256=digest(out/'report.json'))
    (HERE/'GSE76118_EARLY_BIOTYPE_RESULTS.json').write_text(json.dumps(public, indent=2)+'\n')
    append_event(events, 'biotype_audit_finished', status=report['status'], report_sha256=public['report_sha256'])
    print(json.dumps(public))


if __name__ == '__main__':
    main()
