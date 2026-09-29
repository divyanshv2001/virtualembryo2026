"""Consolidate byte-identical small archived model artifacts without removing paths."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


parser = argparse.ArgumentParser(description='Consolidate identical immutable local research artifacts; never remove a file path.')
parser.add_argument('--audit-dir', required=True, type=Path, help='Fresh child directory under E:/virtualembryo-research-storage')
parser.add_argument('--dry-run', action='store_true', help='Hash and plan only; leave artifacts unchanged')
args = parser.parse_args()
private = (Path.cwd() / 'outputs/t1_iterations/private').resolve()
audit_root = Path('E:/virtualembryo-research-storage').resolve()
audit = args.audit_dir.resolve()
if audit.parent != audit_root or not audit.name.startswith('storage_'):
    raise ValueError('Audit must be a fresh storage_* child of E:/virtualembryo-research-storage')
if audit.exists():
    raise ValueError('Preserve previous small-artifact audit')
audit.mkdir(parents=True)
groups = defaultdict(list)
for folder in private.iterdir():
    if not folder.is_dir() or folder.is_symlink() or folder.resolve().drive.lower() != private.drive.lower():
        continue
    if 'prepared' in folder.name:
        continue
    report = folder / 'report.json'
    if not report.exists() or json.loads(report.read_text()).get('status') != 'completed':
        continue
    for path in folder.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.npy', '.npz', '.pt'}:
            continue
        size = path.stat().st_size
        if 1_000_000 <= size < 100_000_000:
            groups[size].append(path)

proposals = []
hashed = 0
for size, paths in sorted(groups.items()):
    if len(paths) < 2:
        continue
    candidates = defaultdict(list)
    for path in paths:
        with path.open('rb') as handle:
            prefix = handle.read(65536)
            handle.seek(max(0, size - 65536))
            suffix = handle.read(65536)
        candidates[hashlib.sha256(prefix + suffix).digest()].append(path)
    for peers in candidates.values():
        if len(peers) < 2:
            continue
        exact = defaultdict(list)
        for path in peers:
            exact[digest(path)].append(path)
            hashed += 1
        for sha, matches in exact.items():
            canonical = matches[0]
            for duplicate in matches[1:]:
                if os.path.samefile(canonical, duplicate):
                    continue
                proposals.append({'canonical': str(canonical), 'duplicate': str(duplicate), 'sha256': sha, 'bytes': size,
                                  'duplicate_mtime_ns': duplicate.stat().st_mtime_ns})

plan = {'created_utc': now(), 'scope': 'Completed local runs only; .npy/.npz/.pt from 1MB to <100MB. No prepared data, source datasets, secrets, active runs or E junctions.',
        'files_hashed': hashed, 'proposals': proposals, 'logical_duplicate_bytes': sum(x['bytes'] for x in proposals)}
(audit / 'plan.json').write_text(json.dumps(plan, indent=2))
if args.dry_run:
    report = {'completed_utc': now(), 'status': 'dry_run', 'files_hashed': hashed,
              'duplicate_paths_that_could_be_consolidated': len(proposals),
              'logical_duplicate_bytes': plan['logical_duplicate_bytes'],
              'unique_files_deleted': 0, 'logical_paths_removed': 0,
              'audit_directory': str(audit)}
    (audit / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    raise SystemExit(0)
before = shutil.disk_usage(private.drive + '/').free
events = audit / 'events.jsonl'
for row in proposals:
    canonical = Path(row['canonical'])
    duplicate = Path(row['duplicate'])
    if not canonical.resolve().is_relative_to(private) or not duplicate.resolve().is_relative_to(private):
        raise ValueError('Candidate outside private workspace')
    if canonical.drive.lower() != duplicate.drive.lower():
        raise ValueError('Cross-volume hardlink')
    if canonical.stat().st_size != row['bytes'] or duplicate.stat().st_size != row['bytes'] or duplicate.stat().st_mtime_ns != row['duplicate_mtime_ns']:
        raise ValueError('Archived artifact changed since hash audit')
    if digest(canonical) != row['sha256'] or digest(duplicate) != row['sha256']:
        raise ValueError('Archived artifact content changed')
    temporary = duplicate.with_name(duplicate.name + '.dedup-link')
    if temporary.exists():
        raise ValueError('Unexpected preexisting dedup temporary')
    try:
        os.link(canonical, temporary)
        if not os.path.samefile(canonical, temporary):
            raise ValueError('Temporary link verification failed')
        os.replace(temporary, duplicate)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not os.path.samefile(canonical, duplicate):
        raise ValueError('Consolidated file verification failed')
    with events.open('a') as handle:
        handle.write(json.dumps({'timestamp_utc': now(), 'canonical': row['canonical'], 'duplicate': row['duplicate'],
                                 'sha256': row['sha256'], 'bytes': row['bytes']}) + '\n')

after = shutil.disk_usage(private.drive + '/').free
report = {'completed_utc': now(), 'status': 'completed', 'files_hashed': hashed,
          'duplicates_consolidated': len(proposals), 'logical_duplicate_bytes': plan['logical_duplicate_bytes'],
          'D_free_before_bytes': before, 'D_free_after_bytes': after, 'observed_free_change_bytes': after - before,
          'verified_all_paths_samefile': all(os.path.samefile(x['canonical'], x['duplicate']) for x in proposals),
          'unique_files_deleted': 0, 'logical_paths_removed': 0, 'audit_directory': str(audit),
          'limitation': 'This consolidates only byte-identical completed-run artifacts; it does not delete the previously blocked forecast arrays or remove unique evidence.'}
(audit / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
