"""Read-only scientific source inventory; generated outputs are excluded."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
records = []
for path in sorted(ROOT.rglob('*')):
    rel = path.relative_to(ROOT)
    if not path.is_file() or any(p in {'.git', '__pycache__', 'outputs'} for p in rel.parts):
        continue
    raw = path.read_bytes()
    row = {'path': rel.as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if path.suffix == '.py':
        tree = ast.parse(raw.decode('utf-8'))
        row['functions_classes'] = [{'name': n.name, 'line': n.lineno, 'kind': type(n).__name__}
                                    for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        row['imports'] = [{'module': n.module, 'line': n.lineno} if isinstance(n, ast.ImportFrom)
                          else {'modules': [a.name for a in n.names], 'line': n.lineno}
                          for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    records.append(row)
missing = ['PLAN.md', 'EXPERIMENTS.md', 'MULTI_AGENT_AUDIT.md', 'experiments/registry.json',
           'src/common.py', 'src/scoring.py', 'outputs/baselines', 'requirements.txt', 'pyproject.toml']
payload = {'scope': str(ROOT), 'source_files': records,
           'absent_expected_paths': [p for p in missing if not (ROOT / p).exists()],
           'data_files': [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.h5ad')],
           'note': 'Nested Git metadata, bytecode and macOS metadata are not scientific results. No historical results reconstructed.'}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'inventory.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(f'Inventoried {len(records)} files; h5ad files: {len(payload["data_files"])}')
