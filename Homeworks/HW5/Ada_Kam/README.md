# HW 5 Documentation

### Setup and run (from `Homeworks/HW5/Ada_Kam/`, Python 3.12)
```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
python -m pytest -q -s && python -m ruff check . && python -m ruff format --check .
docker build -t mfe-hw5:local . && docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```
- The model loaded from `artifacts/model.joblib`, with `course_data.py` 
- To regenerate it, run `uv run --locked python Homeworks/HW5/Ada_Kam/export_model.py` from the repo root; this needs the gitignored HW4 parquet file.

### Input contract (`POST /predict`)
- 12 fields
- Features: 


| Field | Units | Allowed range |
|---|---|---|
| `rsi_21` | 21-day RSI | 0 to 100 |
| `dma_20` | close / 20-day moving average − 1, fraction | −1 to 100 |
| `dma_50` | close / 50-day moving average − 1, fraction | −1 to 100 |
| `dma_100` | close / 100-day moving average − 1, fraction | −1 to 100 |
| `momentum_252_21` | 12-month return excluding the last month, fraction | −1 to 100 |
| `volatility_63` | 63-day annualized volatility, fraction | 0 to 50 |
| `roe` | TTM net income / equity | −1000 to 1000 or null |
| `debt_to_equity` | total debt / equity | −1000 to 1000 or null |
| `book_to_px` | equity / market cap | −1000 to 1000 or null |
| `earnings_yield` | TTM net income / market cap | −1000 to 1000 or null |
| `fcf_yield` | TTM free cash flow / market cap | −1000 to 1000 or null |
| `ebitda_ev` | TTM EBITDA / enterprise value | −1000 to 1000 or null |

`null` means "not reported"

### Output contract

| Field | Type | Meaning |
|---|---|---|
| `prediction` | finite float | Cross-sectional **rank score**, roughly −0.5 to 0.5 |
| `model_version` | string | `"hw4-hgb-rank-v1"` |

- `prediction` is a rank score, not a return. HW4 model was trained on `target_rank = (same-date percentile rank of the 90-day forward return) − 0.5`
- Higher means the stock is expected to rank higher among that month's universe: +0.5 ≈ top of the cross-section, 0 ≈ median, −0.5 ≈ bottom

Example: `example_request.json` (symbol A, 2026-08-31) → `{"prediction": 0.016113780140667493, "model_version": "hw4-hgb-rank-v1"}`

### Design Choices
1. Fundamental features can be None, Technical features cannot
    - For missing fundamental features, has to be explicitly set as Null
2. Bounds for technical features
    - 0 <= RSI <= 100, -1 <= Returns
3. Handle error 500 returns in requests where NaN cannot be written out as JSON

### Model served
`hw4-hgb-rank-v1`: the final HW4 model (`03_backtest.ipynb` cell 2), QuantileClipper → median imputer → scaler → HistGradientBoostingRegressor, trained 2010–2015 on the same-date return rank
`export_model.py` refits it with identical settings (`random_state=42`) and records a reference
input/prediction in `artifacts/model_metadata.json`; the valid-input test checks the API against it (abs 1e-9).

### Lint
- `ruff check .` → All checks passed! ·
- `ruff format --check .` → 5 files already formatted
- Latency budget: test asserts p95 < 250 ms, >10× the measured p95, passes on slower machines but catches regressions like reloading the model per request.

### Test Results
- In-process TestClient latency, n=50 after 5 warm-ups: median=24.3 ms, p95=30.5 ms (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, Windows, Python 3.12.0)
- -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
19 passed, 1 warning in 16.55s

### Docker Results
- More in container_transcript.txt
- Valid Input
```
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@example_request.json"
HTTP/1.1 200 OK
date: Mon, 16 Nov 2026 16:16:34 GMT
server: uvicorn
content-length: 69
content-type: application/json

{"prediction":0.016113780140667493,"model_version":"hw4-hgb-rank-v1"}
```

- Invalid Input (rsi < 0)
```
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@bad_example_request.json"-winter\Homeworks\HW5\Ada_Kam> 
HTTP/1.1 422 Unprocessable Entity
date: Mon, 16 Nov 2026 16:19:08 GMT
server: uvicorn
content-length: 117
content-type: application/json

{"detail":[{"loc":["body","rsi_21"],"msg":"Input should be greater than or equal to 0","type":"greater_than_equal"}]}
```

### References: 
- Lecture 9 demo (app.py), course_data.py (copied from Lectures/modeling)
- Claude was used to generate codes