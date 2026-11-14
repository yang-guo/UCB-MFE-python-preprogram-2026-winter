PYTHON ?= python

.PHONY: test lint format build run
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
