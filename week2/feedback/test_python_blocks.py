"""Execute each handout Python block independently and capture plots/results."""
from pathlib import Path
import contextlib
import io
import json
import traceback
import warnings
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'feedback' / 'python-execution'
OUTPUT.mkdir(exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re

results = []
for source in sorted(ROOT.glob('*.md')):
    blocks = re.finditer(r'```python\s*\n(.*?)```', source.read_text(encoding='utf-8'), re.S)
    for index, match in enumerate(blocks, 1):
        name = f'{source.stem}-block-{index}'
        stdout = io.StringIO()
        entry = {'file': source.name, 'block': index, 'plots': []}
        namespace = {'__name__': '__main__'}
        def capture_show(*args, **kwargs):
            for number in plt.get_fignums():
                target = OUTPUT / f'{name}-figure-{len(entry["plots"])+1}.png'
                plt.figure(number).savefig(target, dpi=120, bbox_inches='tight')
                entry['plots'].append(target.name)
            plt.close('all')
        plt.show = capture_show
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                    exec(compile(match.group(1), f'{source.name}:block-{index}', 'exec'), namespace)
                    if plt.get_fignums():
                        capture_show()
                entry['status'] = 'passed'
            except ModuleNotFoundError as exc:
                entry['status'] = 'blocked_missing_dependency'
                entry['error'] = str(exc)
            except Exception:
                entry['status'] = 'failed'
                entry['error'] = traceback.format_exc()
            entry['warnings'] = [str(item.message) for item in caught]
        entry['output'] = stdout.getvalue()
        plt.close('all')
        results.append(entry)
        print(f'{entry["status"]}: {source.name}, block {index}', flush=True)

report = {
    'python': sys.version,
    'executable': sys.executable,
    'numpy': np.__version__,
    'matplotlib': matplotlib.__version__,
    'counts': {status: sum(r['status'] == status for r in results)
               for status in ['passed', 'failed', 'blocked_missing_dependency']},
    'results': results,
}
(OUTPUT / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report['counts']))
