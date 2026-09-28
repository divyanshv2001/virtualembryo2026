"""Remove trailing empty lines only from generated documents flagged by Git."""
from pathlib import Path
here=Path(__file__).resolve().parent
for name in ('critics/loop_1.md','critics/loop_2.md','experts/replearn.md'):
    p=here/name
    p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n',encoding='utf-8')
