"""Test whether deliberate regressions are caught, using temporary source copies.

No production files are changed. An import/discovery failure is a harness error,
not evidence that the suite detected a behavioral regression.
"""

from pathlib import Path
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile


SOURCE = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ("unreserved_cash_bypass", "backtesting/oms.py",
     "budget > self.available_cash + 1e-8", "budget > self.cash + 1e-8"),
    ("future_price_history", "backtesting/engine.py",
     "self._prices.loc[:signal_date].dropna", "self._prices.dropna"),
    ("mature_label_boundary", "backtesting/engine.py",
     "self._training.label_end.lt(context.signal_date)", "self._training.label_end.le(context.signal_date)"),
    ("always_equal_optimizer", "backtesting/portfolio.py",
     "if len(returns) < self.min_observations or count == 1:", "if True:"),
    ("alpha_sign", "backtesting/analytics.py",
     "excess.mean() - beta * market_excess.mean()", "excess.mean() + beta * market_excess.mean()"),
    ("stale_price_never_stops", "backtesting/engine.py",
     "if position - self._last_seen[symbol] > self.max_stale_sessions:", "if False:"),
]


def main():
    # Keep the venv executable's symlink path: resolving it can select the base
    # interpreter and falsely count missing dependencies as detected mutations.
    python = str(Path(sys.executable).absolute())
    records = []
    expected_count = None
    with tempfile.TemporaryDirectory(prefix="hw4-mutation-audit-") as folder:
        for name, filename, old, new in [("baseline", None, None, None), *MUTATIONS]:
            target = Path(folder) / name
            shutil.copytree(SOURCE / "backtesting", target / "backtesting",
                            ignore=shutil.ignore_patterns("__pycache__", "*.ipynb"))
            shutil.copytree(SOURCE / "tests", target / "tests",
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(SOURCE / "data_loading.py", target / "data_loading.py")
            if filename is not None:
                path = target / filename
                original = path.read_text()
                if old not in original:
                    raise RuntimeError(f"Mutation target changed: {name}")
                mutated = original.replace(old, new, 1)
                ast.parse(mutated)
                path.write_text(mutated)
            result = subprocess.run([python, "-m", "unittest", "discover", "-s", "tests", "-q"],
                                    cwd=target, capture_output=True, text=True, timeout=90)
            count = re.search(r"Ran (\d+) tests", result.stderr)
            if not count or "ModuleNotFoundError" in result.stderr or "Failed to import test module" in result.stderr:
                raise RuntimeError(f"Invalid test harness for {name}: {result.stderr}")
            if name == "baseline":
                if result.returncode:
                    raise RuntimeError(f"Baseline must pass first: {result.stderr}")
                expected_count = int(count.group(1))
            elif int(count.group(1)) != expected_count:
                raise RuntimeError(f"Test discovery changed for {name}")
            records.append(dict(mutant=name, detected=result.returncode != 0,
                                exit_code=result.returncode, test_summary=result.stderr[-1200:]))
            print(name, "BASELINE PASS" if name == "baseline" else
                  ("DETECTED" if result.returncode else "SURVIVED"), flush=True)
    (SOURCE / "audits/mutation_results.json").write_text(json.dumps(records, indent=2) + "\n")
    return int(any(not r["detected"] for r in records[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
