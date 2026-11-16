PYTHON ?= python
DOCKER ?= $(shell command -v docker 2>/dev/null || printf '%s' /Applications/Docker.app/Contents/Resources/bin/docker)

.PHONY: test lint format build run
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff format .
build:
	$(DOCKER) build -t mfe-hw5:local .
run:
	$(DOCKER) run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
