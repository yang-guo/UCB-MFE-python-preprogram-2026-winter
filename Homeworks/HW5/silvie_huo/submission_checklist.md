**Completed application, tests, Dockerfile, Makefile, and dependency versions**
- Completed application: `app.py`
- Tests: `tests/test_app.py`
- Test result (performance):`tests/perf_report.json`
- Sample model input(test model prediction):`tests/reference_case.json`
- Dockerfile, Makefile: `Dockerfile`,`Makefile`
- Dependency versions: `requirements.txt`

**README with the input/output contract, setup, artifact loading instructions, and commands to reproduce your results****
- See README.md

**Your saved HW4 model and preprocessing, any required custom modules, and `example_request.json`**
- HW4 model is at `/preprocessing/hw4_model.py`
- Preprocessing:
```python
cd /preprocessing
python3 hw4_model.py && hw5_export_model.py
cp -r artifacts/ ..
```
- Required custom modules: `course_data.py`
- Example requests: see `example_request.json` and `example_request_tofail.json`

**Passing test and lint output, plus the latency report**
- Test result (performance):`tests/perf_report.json`
- Other tests and lint output:
```
pytest -q
..............................                                                                                          [100%]
30 passed in 3.04s

ruff check .
All checks passed!

ruff format --check .
3 files already formatted
```
**Evidence of valid/invalid requests to the container and a request after restart**

(docker_env) hw5 % curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
date: (sanitized)
server: uvicorn
content-length: 77
content-type: application/json

{"prediction":0.017038218388550738,"model_version":"hgb_w-20261117-bb59b45b"}%                                                                                       
(docker_env) hw5 % curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request_tofail.json
HTTP/1.1 422 Unprocessable Entity
date: (sanitized)
server: uvicorn
content-length: 886
content-type: application/json

{"detail":[{"loc":["body","momentum_63"],"msg":"Field required","type":"missing"},{"loc":["body","momentum_252_21"],"msg":"Field required","type":"missing"},{"loc":["body","volatility_63"],"msg":"Field required","type":"missing"},{"loc":["body","drawdown_252"],"msg":"Field required","type":"missing"},{"loc":["body","net_margin"],"msg":"Field required","type":"missing"},{"loc":["body","operating_roa"],"msg":"Field required","type":"missing"},{"loc":["body","debt_to_assets"],"msg":"Field required","type":"missing"},{"loc":["body","cash_to_assets"],"msg":"Field required","type":"missing"},{"loc":["body","fcf_to_assets"],"msg":"Field required","type":"missing"},{"loc":["body","accruals_to_assets"],"msg":"Field required","type":"missing"},{"loc":["body","filing_age_days"],"msg":"Field required","type":"missing"},{"loc":["body","sector"],"msg":"Field required","type":"missing"}]}%  
(docker_env) hw5 % 

INFO:     172.17.0.1:60270 - "POST /predict HTTP/1.1" 200 OK
INFO:     172.17.0.1:60174 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [1]

make run
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:app:loaded model hgb_w-20261117-bb59b45b
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     172.17.0.1:58852 - "GET /health HTTP/1.1" 200 OK

**A brief explanation of what would need attention before a real deployment**
- See README.md