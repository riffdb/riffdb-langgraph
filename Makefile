UV ?= uv
PYTHON ?= $(UV) run python
RIFFDB ?= riffdb

.PHONY: bootstrap format lint typecheck test check conformance

bootstrap:
	$(UV) sync

format:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

check: format lint typecheck test
	git diff --exit-code

conformance:
	$(RIFFDB) dev --reset --exec '$(PYTHON) -m riffdb_langgraph "$$RIFFDB_ENDPOINT" "$$RIFFDB_CREDENTIAL_FILE" "$$RIFFDB_DATABASE"'
