IMAGE := mfe-hw5:local
PYTHON ?= python

.PHONY: test lint format build run

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 $(IMAGE)
