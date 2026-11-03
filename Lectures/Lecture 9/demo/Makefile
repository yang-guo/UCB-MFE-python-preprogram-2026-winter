.PHONY: setup test lint format serve notebook build run
setup:
	uv sync --locked
test:
	uv run --locked pytest -q
lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
format:
	uv run --locked ruff format .
serve:
	uv run --locked uvicorn app:app --host 127.0.0.1 --port 8769
notebook:
	uv run --locked jupyter lab ../Lesson_9_Demos.ipynb
build:
	docker build -t mfe-lesson9:classroom-v1 .
run:
	docker run --rm --name mfe-lesson9-demo -p 127.0.0.1:8770:8000 mfe-lesson9:classroom-v1
