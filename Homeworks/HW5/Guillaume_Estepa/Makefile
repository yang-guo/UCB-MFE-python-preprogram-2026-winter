PYTHON ?= python

.PHONY: test lint format serve build run
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff format .
serve:
	$(PYTHON) -m uvicorn app:app --host 127.0.0.1 --port 8000
build:
	docker build -t mfe-hw5:local .
run:
	docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
