.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli==1.9.38 quarto
HOST ?= 127.0.0.1
PORT ?= 4200
PREVIEW_FLAGS ?= --no-browser --host $(HOST) --port $(PORT)
DOCS_PYTHONPATH ?= $(CURDIR)/src

.PHONY: help dev preview render check-docs build clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Start the Quarto preview server with live reload.
	PYTHONPATH="$(DOCS_PYTHONPATH)" $(QUARTO) preview docs $(PREVIEW_FLAGS)

preview: dev ## Alias for dev.

render: ## Render the static docs site into docs/site/.
	PYTHONPATH="$(DOCS_PYTHONPATH)" $(QUARTO) render docs

check-docs: ## Validate docs quietly without touching the workspace.
	@set -eu; \
		DOCS_CHECK_DIR="$$(mktemp -d)"; \
		trap 'rm -rf "$${DOCS_CHECK_DIR:?}"' EXIT HUP INT TERM; \
		rsync -a --exclude='site/' --exclude='.quarto/' --exclude='__pycache__/' docs/ "$$DOCS_CHECK_DIR/docs/"; \
		PYTHONPATH="$(DOCS_PYTHONPATH)" $(QUARTO) render "$$DOCS_CHECK_DIR/docs" --quiet

build: render ## Alias for render.

clean: ## Remove rendered docs output.
	rm -rf docs/site
