# Evidence: export, lint and test output

Recorded on 2026-11-17 (17:24 Asia/Shanghai) in `Homeworks/HW5/Qianhao_Meng`, in the same terminal session and immediately before the `make build` in `evidence/container_requests.md`. Lint and tests ran with the folder's `.venv` (Python 3.12.14, from `requirements.txt`). The same `make test` wrote `../latency_report.md`. Output is copied verbatim with the shell prompt shortened to `$` and the repository path shortened to `<repo>`.

## 1. Export the model

Run with the repository `.venv` and the HW4 code restored into the working tree, as described in the README.

```
$ ( cd ../../.. && .venv/bin/python Homeworks/HW5/Qianhao_Meng/export_model.py )
<repo>/Homeworks/HW4/Qianhao_Meng/data.py:125: UserWarning: Quarantining 1 price rows outside the supplied market calendar; inspect the raw database and delivery audit.
  prices, fundamentals, universe, benchmark = course.load_data(database, end=end, symbols=batch)
Saved hw4-ridge-ranks-20141231-v1: 59,183 rows, shares [0.3136, 0.0, 0.0, 0.6864]
Example A on 2014-12-31 -> 0.041160
```

The warning is HW4's own `data.py` quarantining one price row outside the market calendar, exactly as during HW4. The script only prints these two lines after its checks passed: the fit date is 2014-12-31, the training size and rounded weight shares match the HW4 record, and the reloaded bundle reproduces the recorded reference prediction.

## 2. Lint

```
$ make lint
python -m ruff check .
All checks passed!
python -m ruff format --check .
8 files already formatted
```

## 3. Tests

```
$ make test
python -m pytest -q
...................                                                      [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1
  <repo>/Homeworks/HW5/Qianhao_Meng/.venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
19 passed, 1 warning in 0.81s
```

The warning comes from the pinned fastapi 0.162.0 and starlette versions, not from this code. The 19 tests are the ones described in the README. The latency numbers from the performance test are in `../latency_report.md`.
