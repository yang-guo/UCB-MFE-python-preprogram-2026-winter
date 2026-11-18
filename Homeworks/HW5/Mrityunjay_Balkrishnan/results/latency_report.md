# HW5 latency measurement

- Result: median **3.097 ms**, p95 **3.477 ms**.
- Requests: 30 successful `POST /predict` requests after one warm-up.
- Method: Python `time.perf_counter()` around each request using FastAPI's in-process `TestClient`; p95 uses NumPy's linear percentile method. This is not network or container latency.
- Hardware: Apple M2 MacBook Air, 8 CPU cores (4 performance, 4 efficiency), 8 GB memory.
- Runtime: Python 3.12.9; FastAPI 0.162.0; scikit-learn 1.8.0; NumPy 2.3.5.
- Re-run: activate `.venv`, then run `python -m pytest -q -s tests/test_app.py::test_predict_performance`.
