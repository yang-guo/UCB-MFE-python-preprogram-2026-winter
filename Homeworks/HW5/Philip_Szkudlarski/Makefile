PYTHON ?= python

.PHONY: test lint format build run serve latency demo export
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

# Extras (not required by the template)
serve:  # local, no Docker
	$(PYTHON) -m uvicorn app:app --host 127.0.0.1 --port 8000
latency:  # real HTTP against a running service on port 8000
	$(PYTHON) scripts/http_latency.py http://127.0.0.1:8000 200
demo:  # build + run + valid/invalid request + restart; needs Docker
	scripts/container_demo.sh | tee evidence/container_transcript.txt
export:  # offline: re-create artifacts/ from HW4 (needs the repo uv env + HW4 database)
	cd ../../.. && .venv/bin/python Homeworks/HW5/Philip_Szkudlarski/export_hw4_model.py
