# Homework 5: Serve and containerize a model

**Due Monday, November 16, 2026, at 11:59 p.m. Pacific (PST).** This is two weeks after the November 2 productionization lesson. The course late-day policy applies.

Submit a pull request with your work in `Homeworks/HW5/<First_Last>/` and assign `kevinramlal` and `raghuramkowdeed`.

## Start here

Copy the contents of [template](template/) into your submission folder, including the dotfiles. Complete the TODOs in your copy. The five milestones retain the previous course's assignment structure.

**Prerequisite: complete [HW4](../../Lectures/Lecture%208/hw4.md). HW5 serves the fitted stock-return model and learned preprocessing that you used in your HW4 experiment.** Preserve its feature definitions, ordering, units, and prediction horizon.

Use Python 3.12 and a separate environment for this assignment. Before installing dependencies, update the model-library versions in `requirements.txt` to match your HW4 training environment. From your submission folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell. If `make` is unavailable, run the equivalent commands listed below.

Serialize your HW4 fitted model together with its learned preprocessing. If it uses custom transformers, include their importable Python modules in the service and container. Document which HW4 model version you are serving and include an example input from that experiment.

Match the training versions in your serving environment and Docker image. Loading sklearn artifacts across versions is unsupported. Include the fitted artifact when small enough for Git, or document an agreed course-accessible location for a large artifact. The service loads this artifact without retraining or querying the HW4 database. Load only artifacts you created or trust.

## Milestone 1: Model serialization and API design

Implement a FastAPI `POST /predict` endpoint:

- Define `InputData` and `OutputData` Pydantic models. Document the exact feature names/order, types, units, and allowed ranges. You may use named fields or a fixed-length feature list.
- Load the fitted model and its learned preprocessing once per server process, using the provided lifespan function. Do not fit or refit on incoming requests.
- Return **200** with a finite numeric `prediction` and a `model_version` for valid input. Preserve the same input representation used during training, including DataFrame column names if required.
- Return **422** for missing required fields, invalid types, extra fields, non-finite values, and the wrong feature count. Define these constraints explicitly. A generic `list[float]` does not automatically check the model's required width; a width error raised later by sklearn would otherwise become a server error.
- Return **500** for an unexpected inference failure. Do not turn inference errors into a successful prediction. Keep detailed exceptions in server logs rather than the response.

Set and justify input constraints appropriate to your HW4 features. Numeric strings and booleans should not silently become numeric features; declare any categorical fields separately.

Run locally with `python -m uvicorn app:app --host 127.0.0.1 --port 8000`. The interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Milestone 2: Testing

Complete the four test groups in `tests/test_app.py`, adding cases where needed:

1. **Valid prediction:** use your saved HW4 model, assert 200 and the response schema, and compare the API prediction with a direct pipeline call. Also retain an input and expected prediction recorded from your HW4 code before wrapping it in the API. Explain where that reference result came from, and use an appropriate numerical tolerance.
2. **Input validation:** exercise the invalid cases above and assert 422. Show that invalid input does not reach inference.
3. **Inference failure:** use valid input and deliberately replace the loaded model's `predict` method with one that raises. Assert 500. Mocking is appropriate for this failure test; use the real model for the positive test.
4. **Performance:** warm up the endpoint, measure at least 20 successful requests with `time.perf_counter`, and report median and p95 latency with your hardware and measurement method. State whether this is an in-process `TestClient` measurement or real localhost HTTP. A universal 500 ms cutoff is not required; if you assert a budget, justify it for your environment.

Use reusable fixtures. The provided clients use `with TestClient(...)` so startup/shutdown runs. The failure client uses `raise_server_exceptions=False` so you can inspect an HTTP 500 instead of having the client re-raise the exception. A server can return a plain-text 500; do not require a JSON error body unless you implement that contract.

Run `make test` or `python -m pytest -q`. The initial TODO failures are intentional; replace them with real assertions. Do not leave empty, skipped, or always-passing tests in the submission.

## Milestone 3: Containerization

Complete the supplied Dockerfile using `python:3.12-slim`. Set `/app` as the working directory, install requirements, and copy the application, custom transformer modules, and saved HW4 artifact into the image. Keep local environments, databases, and credentials out of the build context.

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The process listens on port 8000 inside the container. Stop any other demo using host port 8000 first, or choose another host port. Document how another person can reproduce the build.

## Milestone 4: Automation, tooling, and readability

The supplied Makefile has `test`, `lint`, `format`, `build`, and `run` commands. Keep them working for your submission.

```sh
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Ruff replaces the previous assignment's separate Black/flake8/isort tools to match this lesson. Type checking is optional. Use clear names and keep reusable model logic separate from request handling.

**Optional CI extension:** run these checks in GitHub Actions on your fork. Adapt the [lesson workflow](../../.github/workflows/lesson-9.yml), pointing its working directory and install command at your submission. Include the run link if you do this. CI is not an additional requirement to pass.

## Milestone 5: Make a request to the container

Send a real request using curl, the `/docs` page, Postman, or a Python HTTP client. Save a valid input matching your HW4 feature contract in `example_request.json`, then run:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Save a screenshot or a short terminal transcript showing the request, 200 response, prediction, and model version. Also show a rejected invalid request. Stop and restart the same image, then demonstrate that the model remains available.

## Submission checklist

- [ ] Completed application, tests, Dockerfile, Makefile, and dependency versions
- [ ] README with the input/output contract, setup, artifact loading instructions, and commands to reproduce your results
- [ ] Your saved HW4 model and preprocessing, any required custom modules, and `example_request.json`; document access if the artifact is too large for Git
- [ ] Passing test and lint output, plus the latency report
- [ ] Evidence of valid/invalid requests to the container and a request after restart
- [ ] A brief explanation of what would need attention before a real deployment

Do not commit `.venv`, databases, credentials, or large generated artifacts. You may consult the [lesson demo](../../Lectures/Lecture%209/demo/) and use AI tools. Cite adapted code and explain your own implementation and checks. Serve your HW4 model; the lesson's synthetic demo model is not a substitute.

**Pass/no-pass:** correct API behavior, meaningful tests, a reproducible container, readable code, and the requested evidence. Model investment performance is not graded in this assignment.

## References

- [Lesson 9 slides, demos, reading, and video](../../Lectures/Lecture%209/README.md)
- [FastAPI lifespan in tests](https://fastapi.tiangolo.com/advanced/testing-events/)
- [Starlette TestClient and 500 responses](https://starlette.dev/testclient/)
- [sklearn model persistence and version compatibility](https://scikit-learn.org/stable/model_persistence.html)
