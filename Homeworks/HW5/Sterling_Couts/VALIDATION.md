# HW5 requirement checklist and evidence

Authority: [hw_5.md](../hw_5.md). Verification completed November 16, 2026.
All five technical milestones are verified. Commit, push and pull-request
submission remain separate actions after review.

| Requirement | Implementation and observed evidence | Status |
|---|---|---|
| Completed HW4 model and learned preprocessing | Unchanged fitted artifact and 209 previously recorded reference rows; metadata identifies HW4 commit `9d94d05` | Verified |
| Separate Python 3.12 environment; matching model-library versions | `requirements.txt`, [environment](evidence/environment.json); Python 3.12.9 locally; version checks also passed inside Docker | Verified |
| Exact names/order, types, units, ranges and prediction horizon | `schemas.py`, `serving.py`, README contract; reversed-field-order and boundary tests | Verified |
| Load once using lifespan; no fitting or database access | `app.py`; lifespan, forbidden-fit and model-state tests | Verified |
| 200, finite prediction and model version | Real-model API/direct comparison; all 209 saved HW4 predictions agree within absolute `1e-12` | Verified |
| 422 for missing/extra fields, invalid types, wrong width, non-finite values | Parametrized tests prove invalid requests never invoke inference | Verified |
| 500 for unexpected inference failure, detailed logs only | Forced `predict` failure, generic response, request ID and server-log assertions | Verified |
| Reusable fixtures and all four required test groups | `tests/test_app.py`; [126 passing tests](evidence/tests.txt) | Verified |
| Warmed performance measurement with hardware/method | 5 warmups, 100 sequential measured requests per method; results below | Verified |
| Python 3.12 slim image, correct modules/artifact, restricted build context | `Dockerfile`, `.dockerignore`; [build output](evidence/docker_build.txt); image contains exactly eight application files | Verified |
| Test/lint/format/build/run commands | Supplied Makefile retained; commands executed through documented equivalents; [lint](evidence/lint.txt), [format](evidence/format.txt) | Verified |
| Real valid and invalid container requests | [HTTP transcript](evidence/container_requests.txt) records request bodies, 200/422 responses, prediction and version | Verified |
| Stop/restart same image and repeat prediction | [Container verification](evidence/container_verification.json): identical image ID, matching internal artifact SHA256 and successful request after restart | Verified |
| README, artifact loading, reproduction, deployment considerations and attribution | README; adapted sources cited and implementation/checks explained | Verified |
| Exclude environments, databases, credentials and large generated artifacts | `.gitignore`, Docker allowlist and final Git inventory in `submission_files.txt` | Verified |
| PR under `Homeworks/HW5/Sterling_Couts/`, assign both graders | Branch `hw_5_Sterling_Couts`; assign `kevinramlal` and `raghuramkowdeed` after review | Not started |

## Observed latency

Each measurement uses `time.perf_counter`, 5 warmup requests, then 100 successful
sequential requests (concurrency 1). p95 uses NumPy's linear percentile method.
Raw timings and method descriptions are retained in the linked JSON files.

| Method | Median (ms) | p95 (ms) | Evidence |
|---|---:|---:|---|
| In-process TestClient, no network socket | 4.677 | 8.230 | [Recorded timings](evidence/latency_testclient.json) |
| Real localhost HTTP to separate Uvicorn process | 6.252 | 7.335 | [Recorded timings](evidence/latency_localhost.json) |
| Real localhost HTTP to Docker container | 4.767 | 7.260 | [Recorded timings](evidence/latency_docker.json) |

Hardware: Intel Core i5-1335U, 10 physical cores / 12 logical processors,
approximately 16 GB RAM, Windows 11 Pro. Docker uses the WSL 2 Linux backend.
See [hardware details](evidence/hardware.json). Measurements occurred at different
times on an active desktop; differences do not establish a container speedup.
These are warmed single-client measurements, not startup or concurrent-load tests.

## Artifact and container identity

The fitted model and original reference CSV are byte-for-byte copies from HW4.
The custom estimator classes retain the same Python syntax trees; import
formatting is the only source change. No HW4 training or data processing runs
as part of this service.

Model artifact SHA256:

```text
84e526d287926f77a4ef478a9923a39e49638ba112adb7ca43d6fa0ccfa1693a
```

The saved AAT example produces `0.03153401217040266` locally and in the
container, matching the prior HW4 result `0.0315340121704026`. Container startup
also verifies the six model-library versions. The image runs as user
`10001:10001`. Its image ID, internal model hash and exact application file list
are recorded in the container verification file. The temporary test container
was removed after verification; the locally built image remains available.

The test run emits one Starlette deprecation warning about its httpx TestClient
adapter. It does not affect the 126 passing tests. The pinned course-compatible
dependency combination is retained; a future migration should be tested separately.

`evidence/final_verification.json` records the final scope, provenance and
documentation checks. No files have been staged, committed or pushed for HW5.
