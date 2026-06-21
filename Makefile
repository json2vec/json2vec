.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli quarto
HOST ?= 127.0.0.1
PORT ?= 4200
PREVIEW_FLAGS ?= --no-browser --host $(HOST) --port $(PORT)

.PHONY: help dev preview render build clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Start the Quarto preview server with live reload.
	$(QUARTO) preview $(PREVIEW_FLAGS)

preview: dev ## Alias for dev.

render: ## Render the static docs site into site/.
	$(QUARTO) render

build: render ## Alias for render.

clean: ## Remove rendered docs output.
	rm -rf site
