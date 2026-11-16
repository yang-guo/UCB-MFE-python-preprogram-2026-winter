PYTHON ?= python

.PHONY: test lint format build run evidence
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
evidence:
	{ $(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check . && $(PYTHON) -m pytest -q -s; } 2>&1 | tee evidence/test_and_lint.txt
	bash evidence/container_demo.sh
