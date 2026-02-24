PYTHON ?= python

.PHONY: test lint format

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
