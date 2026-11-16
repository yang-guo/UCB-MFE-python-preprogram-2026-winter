# HW5 – Philip Szkudlarski: serving my HW4 model

This service serves the **final model from my HW4** (run 10, `pairs+fcf+revenue+momentum [trailing 126d]`) behind a tested FastAPI `POST /predict` endpoint. The repo also includes a Dockerfile for it.

| | |
|---|---|
| Served model | HW4 final model, frozen at the **2026-09-30 close** (last HW4 signal date) |
| `model_version` | `hw4-run10-trailing126d-2026-09-30-v2` (v2 added the informational expected-return output; scores are identical to v1) |
| Artifact | [`artifacts/hw4_model.joblib`](artifacts/hw4_model.joblib) (3 KB, in Git): a fitted sklearn `Pipeline` plus version and feature metadata |
| Metadata | [`artifacts/model_metadata.json`](artifacts/model_metadata.json): signal parameters, weights, library versions, parity checks |
| Example input | [`example_request.json`](example_request.json): HW4's #1-ranked stock on 2026-09-30 (MU) |
| Training environment | Python 3.12.9, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3 (repo `uv.lock`); pinned identically in `requirements.txt` and the image |

## 1. What the model is and how it was serialized

HW4's model is a closed-form ranking model. On each decision day $t$, each of four buy signals $s$ flags some stocks. The stock's score is:

$$\text{score}_i(t)=\sum_s w_s(t)\,p_{i,s}(t)\;+\;\#\{s:\ i \text{ flagged by } s,\ w_s(t)>0\}$$

- $p_{i,s}$ is the stock's percentile among the names that signal $s$ flagged that day.
- $w_s$ are the trailing-126-day weights. Each is the positive part of the signal's mean *finished* 90-day excess return, normalized to sum to 1.
- HW4 buys the top-scored names. A higher score means buy first.

Both parts are learned from HW4 data. I packaged them as a scikit-learn `Pipeline` whose two steps are custom estimators in [`hw4_model.py`](hw4_model.py):

1. **`FlaggedPercentileTransformer`** is the learned preprocessing.
   - It stores, for each signal, the 2026-09-30 cross-section of flagged strengths: 16 pairs, 128 FCF, 130 revenue and 49 momentum names.
   - It maps a raw strength to the percentile HW4 computed (pandas `rank(pct=True)`, average ties).
   - An unseen value is ranked as if it joined that day's cross-section.
2. **`SignalScoreRegressor`** holds the frozen weights $w_s$ and HW4's trailing expected excess returns $\hat\mu_s$. It computes the score, and also the informational `expected_excess_return_90d` (see section 2).

[`export_hw4_model.py`](export_hw4_model.py) creates the artifact. It is an offline step and is not part of the image.
- It imports my **unchanged HW4 code** (`Homeworks/HW4/Philip_Szkudlarski/hw4.py`) and rebuilds the four signal matrices the way `hw4.main()` does: the same parameter choices, waiting periods and walk-forward FCF sleeve.
- It computes the weights with `hw4.trailing_strategy_weights`, then fits and saves the pipeline.
- **Nothing was re-tuned.** No test-period result was used to change the HW4 model.
- It asserts three parity checks before saving. All three passed:

| Check | Result |
|---|---|
| `pipeline.predict` vs HW4's own `combine_weighted` score, all 157 candidates on 2026-09-30 | max abs diff **2.2e-7** (HW4 computes in float32) |
| Weights implied by my recomputed $\hat\mu_s$, i.e. $\max(\hat\mu_s,0)/\sum$, vs HW4's weights on 2026-09-30 | max abs diff **2.2e-8** |
| Yearly mean weights vs HW4 `outputs/test_dynamic_weights_by_year.csv` | max abs diff **2.3e-8** |
| Every stock HW4 actually bought in the test period has a finite rebuilt score on its signal day | **964 / 964** |

**State of the model on 2026-09-30.** All four inputs are part of the contract. On this date, though, the trailing weights are momentum **0.9915**, FCF **0.0085**, pairs **0** and revenue **0**: pairs and revenue had negative trailing excess returns. Only momentum and FCF flags therefore make a stock a candidate. HW4 behaves the same way: its weights move daily, and in 2026 momentum dominated (README of HW4).

**Artifact loading.** The service loads `artifacts/hw4_model.joblib` once per process in the FastAPI lifespan. The path is resolved relative to `app.py`. `hw4_model.py` must be importable because joblib stores the class paths. The service never retrains and never touches the HW4 database.
- **Security:** joblib/pickle can execute code on load, so the service loads only this artifact, which I created myself (see the sklearn persistence docs).
- **To re-create it** you need the repo's uv environment and `equities_top2000.duckdb` in `Lectures/modeling/data/`. You also need HW4's git-ignored `outputs/` folder: the parity checks read `test_dynamic_weights_by_year.csv` and `test_fills.csv`, so run `hw4.py` once first. Then run `make export`, about 3 minutes; it rewrites `artifacts/` and `example_request.json`. The service and the tests need none of this; they use only the committed artifact.

## 2. API contract (Milestone 1)

`POST /predict`, JSON object, **exactly these four keys, all required**, in training order. Each value is a number, or `null` when that signal did **not** flag the stock:

| # | Field | HW4 signal / units | Allowed |
|---|---|---|---|
| 1 | `pairs_divergence_z` | \|z-score\| (standard deviations) of the 180-day log price-ratio spread by which the stock lags a same-industry peer with correlation ≥ 0.7 and beta difference ≤ 0.1; EPS intact | `null` or 1.5 – 10 |
| 2 | `fcf_quality_score` | Mean same-day percentile (0–1) of FCF growth, FCF/share growth, share reduction and FCF yield; flagged when TTM FCF growth ≥ 40% (or FCF turned positive), dilution ≤ 3% and FCF/share up | `null` or (0, 1] |
| 3 | `revenue_growth_vs_industry` | YoY revenue growth minus the industry median, **as a fraction** (0.05 = 5 points), plus HW4's 0.001 offset; flagged with growth accelerating and operating margin > 0 | `null` or 0.001 – 10 |
| 4 | `momentum_percentile` | Percentile (0–1) of the 12-1 month return (252 days, skipping the last 21) in that day's universe; HW4 flags the top 10% | `null` or 0.9 – 1 |

Each value is the signal's strength on the decision date *after* HW4's waiting period (pairs, FCF and revenue: 2 sessions; momentum: 1). The prediction horizon is HW4's: the score ranks stocks for purchase at the **next session's close**, held **90 calendar days**.

**Why these limits.**
- **Lower bounds** are HW4's own flag thresholds. A smaller value cannot come from a flagged stock, so it signals a caller error.
- **Upper bounds:**
  - Percentiles and scores are capped at 1 by definition.
  - z ≤ 10 and a revenue gap ≤ 1,000 points are generous sanity limits. The 2026-09-30 maxima were 3.8 and 3.3.
  - Together they catch unit mistakes, such as `97` sent as a percent instead of `0.97`, and data errors.

**Output** (`200`):

```json
{"prediction": 2.9735976986327786, "is_candidate": true, "expected_excess_return_90d": 0.0697452034600955, "model_version": "hw4-run10-trailing126d-2026-09-30-v2"}
```

- `prediction` is the finite HW4 composite score, in [0, 5]. Candidates score > 1. **`0.0` means not a buy candidate**: no positive-weight signal flagged the stock (HW4 had NaN here).
- `expected_excess_return_90d` is **informational**, as a fraction (0.0697 = +7.0%). It gives a return-type number next to the ranking score and does not change it.
  - **Definition:** $\sum_s w_s\hat\mu_s / \sum_s w_s$ over the positive-weight signals that flag the stock. $\hat\mu_s$ is HW4's trailing-126-day mean 90-day return of the stocks signal $s$ flagged, in excess of the same-day universe average (not of ^SP500TR), using finished outcomes only.
  - **It is a historical average per signal, not a forecast for each stock.** On 2026-09-30 it takes only three values: momentum only **+7.03%** (29 stocks), momentum + FCF **+6.97%** (20), FCF only **+0.06%** (108). Within a group, the score does the ranking.
  - `null` for non-candidates.
- Every response carries an `X-Request-ID` header, which also appears in the server log line of a failure.

**Status codes:**

| Code | When |
|---|---|
| **200** | Valid input. |
| **422** | Missing key, extra key, wrong feature count (a list or a 3/5-key object), numeric string, boolean, `NaN`/`Infinity`, out-of-range value, malformed JSON. Pydantic runs in `strict=True`, `extra="forbid"`, `allow_inf_nan=False`. The 422 body lists `loc`/`msg`/`type` and does **not** echo the input (see the bug below). |
| **500** | `{"detail": "Prediction unavailable"}` when inference raises or returns a non-finite value. The traceback goes only to the server log. |

`GET /health` returns the loaded version and snapshot date. The interactive docs are at `http://127.0.0.1:8000/docs`.

**Bug the tests found.** At first, a `NaN`/`Infinity` input returned **500**, not 422. FastAPI's default 422 handler echoes the rejected value, and JSON cannot encode `-inf`. [`app.py`](app.py) now has a validation handler that omits the value, and `test_non_finite_values_rejected` is the regression test.

## 3. Setup and commands

From this folder (macOS/Linux; on Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make lint        # python -m ruff check . && python -m ruff format --check .
make test        # python -m pytest -q
make serve       # python -m uvicorn app:app --host 127.0.0.1 --port 8000
make build       # docker build -t mfe-hw5:local .
make run         # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
make demo        # Milestones 3+5 end to end in Docker, saves evidence/container_transcript.txt
make latency     # real-HTTP latency against whatever runs on port 8000
```

**Reproducing the image.**
- **Inputs:** everything the build needs is in this folder — `requirements.txt` (exact pins), `app.py`, `hw4_model.py` and `artifacts/`.
- **Build context:** `.dockerignore` keeps `.venv`, caches, `.env*`, databases, tests and the export script out of it.
- **Image:** it runs as a non-root user and listens on port 8000.
- **Steps:** another person needs only Docker. Run `make build && make run` from this folder; stop any other service on host port 8000 first, or change the host side of `-p`.
- **Version:** record the image digest, which `make demo` prints, together with `model_version`.

## 4. Tests (Milestone 2) – `tests/test_app.py`, 39 passing

**Reference results.** `export_hw4_model.py` wrote [`artifacts/hw4_reference_scores.csv`](artifacts/hw4_reference_scores.csv) before any API code existed.
- It records all 157 candidates on 2026-09-30: their inputs, the HW4 score from `hw4.combine_weighted`, and the HW4 rank.
- **Tolerances:** HW4 scores in float32, so API-vs-HW4 comparisons use `abs=1e-5`. API-vs-direct-pipeline comparisons share one float64 code path and use `1e-12`.
- **Hand check of the top row (MU):**
  - FCF percentile 35/128 = 0.2734 and momentum percentile 48/49 = 0.9796.
  - 0.00849·0.2734 + 0.99151·0.9796 + 2 agreeing signals = **2.9736**, which matches the API.

| Group | Tests |
|---|---|
| 1. Valid prediction | Real model; 200 and exact schema; API == direct `pipeline.predict` on the training-format DataFrame; API == HW4 recorded score for the example and for 6 rows spread across the ranking (ranks 1, 2, 6, 26, 61, last); all 157 rows reproduce HW4 scores **and ranks**; row-alone == row-in-batch (frozen preprocessing, slide 41); `expected_excess_return_90d` == the value computed in the export directly from HW4's arrays (all 157 rows, plus the 6 API rows), == MU by hand from `model_metadata.json`, and == $\hat\mu_{mom}$ for a momentum-only stock; all-null input → 0.0 / not a candidate / null return; `/health` version |
| 2. Input validation | 19 invalid payloads plus NaN / Infinity / −Infinity and malformed JSON, all **422**. A fixture replaces `pipeline.predict` with `pytest.fail`, so these tests also prove invalid input never reaches inference |
| 3. Inference failure | Valid input with `predict` monkeypatched to raise → **500**, generic body, no exception text, request ID present. A model returning NaN → 500, not a "successful" NaN |
| 4. Performance | 10 warm-up requests, then 200 timed with `time.perf_counter`; median and p95 are printed |

**Latency report** (Apple M5, 16 GB, macOS 26.5.2, Python 3.12.9, 2026-11-16):

| Measurement | Requests | Median | p95 |
|---|---|---|---|
| In-process `TestClient` (pytest; no network) | 200 after 10 warm-up | 0.80 ms | 0.89 ms |
| Real HTTP to the **Docker container** (`-p 127.0.0.1:8000:8000`, Docker Desktop's Linux VM), one keep-alive `httpx` client, sequential ([`scripts/http_latency.py`](scripts/http_latency.py)) | 200 after 20 warm-up | 1.00 ms | 1.37 ms |

**Budget:** the test asserts **p95 < 50 ms** for the in-process measurement.
- That is about 50× headroom over this laptop, so a slower laptop or CI runner won't fail it.
- It still catches regressions such as reloading the artifact or refitting on every request.
- It is not a production SLO. The real budget depends on the caller: a daily batch rebalance needs throughput, not milliseconds.

Full output: [`evidence/test_and_lint_output.txt`](evidence/test_and_lint_output.txt).

## 5. Requests to the running service (Milestone 5)

**Container run:** [`evidence/container_transcript.txt`](evidence/container_transcript.txt) (Docker Desktop 4.99.0, Engine 29.9.3, 2026-11-16). It was produced by:

```sh
make demo   # = scripts/container_demo.sh | tee evidence/container_transcript.txt
```

The transcript shows each step:
1. `docker build -t mfe-hw5:local .` builds image `sha256:9ff75e49e4d5…`.
2. `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` starts it.
3. `curl -i` with `example_request.json` returns **200 OK** with `prediction` 2.9735976986327786, `is_candidate` true, `expected_excess_return_90d` 0.0697, `model_version` `hw4-run10-trailing126d-2026-09-30-v2`, and an `x-request-id` header.
4. A rejected request (numeric string `"0.6"` plus `97` sent as a percent) returns **422**, with both fields named.
5. Real-HTTP latency against the container: **200** requests.
6. `docker stop`: the service is down, and `--rm` removed the container.
7. The **same image** is started again and returns the **same prediction**. The packaged model is part of the image, not of a container's state.
8. Inside the image: `appuser` (non-root), Python 3.12.16, sklearn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3 — the same libraries as HW4 training. The patch-level Python difference (3.12.9 → 3.12.16) is within the same minor version. `/app` contains only `app.py`, `hw4_model.py`, `requirements.txt` and the two artifact files.


## 6. Before a real deployment

- **The feature pipeline is outside the service.** It accepts signal strengths that must be computed exactly as in HW4: point-in-time fundamentals (filing date + 1), waiting periods, the top-500 universe and split repair. The most likely production bug is training/serving skew: valid JSON, wrong numbers (slide 42, question 1). A real system would compute features in the same reusable module and test parity per release.
- **The model goes stale.** In HW4 the weights and the flagged cross-section change every day. This artifact is the 2026-09-30 snapshot. A deployment needs a scheduled job to re-export the artifact and promote a new `model_version` after the parity checks pass. It also needs to keep the previous image for rollback.
- **Batch, not online.** HW4 trades once per close, so nightly batch scoring of the whole universe fits better than per-stock HTTP calls (slide 27). Ranking depends on the whole cross-section.
- **Releases and monitoring.**
  - Record the image digest and model version, and verify every instance runs the same one (Knight Capital, slide 4).
  - Monitor error rate and latency, input distributions (share of nulls, range hits) and live 90-day outcomes against the trailing $\hat\mu_s$.
  - Anthropic's postmortem shows how quality regressions can pass noisy evaluations and stay hidden behind healthy-looking infrastructure. Continuous quality checks on production output matter as much as uptime.
- **Failure isolation.** The service has no fallback. If it is down, the trading job should skip new entries rather than buy blindly, with timeouts and retries that are safe to repeat (scoring is read-only). This follows the dependency-isolation theme of the Netflix microservices talk.
- **Security and operations.**
  - Load only signed or trusted artifacts; consider skops instead of pickle.
  - Add authentication and rate limits.
  - Run CI on every PR. I did not set up the optional GitHub Actions run: it needs a push to a fork. The commands are the three in `make lint` / `make test`.
  - Pin the base image by digest.
  - Size workers against memory. The model is tiny, so one copy per worker costs little.

## 7. Sources, reuse and AI use

- **Starter and course material:**
  - HW5 template: `Homeworks/HW5/template`.
  - Lesson 9 demo (`Lectures/Lecture 9/demo/app.py`, `tests/`): I adapted its lifespan loading, request ID, 500 handler and the "invalid input never reaches the model" test pattern.
  - Lesson 9 slides: contract, row-alone vs batch test, releases, monitoring.
- **Documentation:**
  - FastAPI lifespan-in-tests docs: `with TestClient(...)`.
  - Starlette TestClient docs: `raise_server_exceptions=False`.
  - scikit-learn model-persistence guide: version matching, trusted loading, metadata.
- **Further reading:** the Anthropic postmortem and the Netflix talk inform section 6.
- **My HW4 code** is reused unchanged through imports in `export_hw4_model.py`.
- **AI use:** the code and this README were written with an AI assistant (Claude Code) at my direction. I reviewed the contract, the bounds, the parity checks and the test results. The export reproduces HW4 exactly rather than changing the model.
- **Known warning:** pytest shows Starlette's deprecation warning for `httpx` in `TestClient`. `httpx==0.28.1` is the template's pinned version, and it works.
