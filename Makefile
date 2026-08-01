.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli==1.9.38 quarto
HOST ?= 127.0.0.1
PORT ?= 4200
PREVIEW_FLAGS ?= --no-browser --host $(HOST) --port $(PORT)
DOCS_PYTHONPATH ?= $(CURDIR)/src

.PHONY: help dev preview render build clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Start the Quarto preview server with live reload.
	PYTHONPATH="$(DOCS_PYTHONPATH)" $(QUARTO) preview docs $(PREVIEW_FLAGS)

preview: dev ## Alias for dev.

render: ## Render the static docs site into docs/site/.
	PYTHONPATH="$(DOCS_PYTHONPATH)" $(QUARTO) render docs

build: render ## Alias for render.

clean: ## Remove rendered docs output.
	rm -rf docs/site
