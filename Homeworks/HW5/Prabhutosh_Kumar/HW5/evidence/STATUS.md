# Verification status — November 16, 2026

- Real HW4 final fit reproduced with the locked alpha 10 and 139,554 completed training rows.
- Original database SHA256 verified: `a028641d96c12e6f29d5dd34e3a6ebc059074dd38351648bf0d970b1dd4095b9`.
- All 249,796 historical HW4 predictions matched (`rtol=1e-9`, `atol=1e-10`); maximum absolute difference `1.3189319428286161e-16`. See `../artifacts/metadata.json`.
- All 33 API tests pass; see `tests.txt`. One dependency deprecation warning is retained in the output.
- Ruff lint and format checks pass; see `checks.txt`.
- Real localhost HTTP returned 200 for the example and 422 for the wrong-width request; see `localhost.txt`. This is uvicorn on the host, not Docker evidence.
- **Container verification passed:** Docker Desktop built and ran `mfe-hw5:local` on Linux arm64. Valid input returned 200 and prediction `0.030344780713053604`; invalid feature count returned 422; stop/start of the same container returned the identical prediction and model version. See [container.txt](container.txt) for the full build and HTTP transcript. The verification container was removed afterward; the image remains installed.
- Verified image ID: `sha256:84f793599e5733e4647f95481bb610926bc75b16bb0d01de1d6f32ffa0ebdcc6`.

## Latency measurement

Hardware: **Apple M1, 8 GiB RAM**, arm64 Mac running macOS 26.5.2. CPU model and physical RAM were read using `sysctl -n machdep.cpu.brand_string` and `sysctl -n hw.memsize` (8589934592 bytes). Serving Python: 3.12.5; original HW4 training/export Python: 3.12.9. All six model-library versions are identical between training and serving and checked on startup.

Measured with `time.perf_counter` around each complete synchronous TestClient POST: five warmup requests, then 50 sequential successful requests. This includes schema validation, DataFrame construction, pipeline prediction, and JSON response serialization. It excludes actual network and container overhead.

- Median: **2.7696 ms**
- p95: **3.0493 ms**, NumPy percentile with its default linear interpolation
- No performance budget asserted; these are local observations rather than a deployment SLA.
