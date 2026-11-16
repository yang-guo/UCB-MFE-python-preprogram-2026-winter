
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
	docker build -t hw5-stock-api .

run:
	docker run --rm --name hw5-stock-api -p 127.0.0.1:8000:8000 hw5-stock-api
