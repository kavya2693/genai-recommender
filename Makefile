.PHONY: data run demo check test lint

data:
	uv run python scripts/generate_data.py

run:
	uv run genai-reco

# Also drafts three messages with llama3.2 through a local Ollama server.
demo:
	OLLAMA_NUM_GPU=0 uv run genai-reco --llm ollama --demo-n 3

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest -q

check: lint test
