PYTHON ?= python
URL ?= http://127.0.0.1:8000

.PHONY: test lint format build run serve latency
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff format .
build:
	docker build -t mfe-hw5:local .
run:
	docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
# Local (non-Docker) server, and real-HTTP latency against whichever server is running.
serve:
	$(PYTHON) -m uvicorn app:app --host 127.0.0.1 --port 8000
latency:
	$(PYTHON) scripts/measure_latency.py --url $(URL)
