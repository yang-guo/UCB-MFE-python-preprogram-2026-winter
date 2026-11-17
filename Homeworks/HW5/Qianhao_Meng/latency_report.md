# Latency report

- Requests: 50 successful POST /predict after 5 warm-up requests
- Median: 0.93 ms
- p95: 1.05 ms
- Method: in-process FastAPI TestClient, time.perf_counter around each request, no network hop
- Hardware: arm64 arm, 12 logical cores, macOS-26.6.2-arm64-arm-64bit
- Python 3.12.14
