"""Small classroom support utility; report logic stays in the teaching notebooks.

Keep interpreter/kernel plumbing out of the live lesson. The environment change is
limited to the current Python process and its report subprocesses. No user-level
kernel is installed. A kernel restart naturally clears the lookup change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


def prepare_report_run(output_root: Path) -> tuple[Path, str]:
    """Create a fresh output folder and a kernel using this Python interpreter."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='reports_', dir=output_root))
    kernel_root = run_dir / 'jupyter'
    kernel_name = 'course-report'
    kernel_dir = kernel_root / 'kernels' / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Course report',
        'language': 'python',
    }), encoding='utf-8')
    prior_path = os.environ.get('JUPYTER_PATH')
    os.environ['JUPYTER_PATH'] = str(kernel_root) + (os.pathsep + prior_path if prior_path else '')
    return run_dir, kernel_name
