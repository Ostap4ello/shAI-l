# shAI — Architecture Overview

This document briefly describes the shAI codebase for developers: its major packages, responsibilities, and runtime flows. The layout mirrors the per-file/module hierarchical design used across the project.

## Package tree

```
shAI_ostap4ello/src/
  __main__.py
  config/
  db/
  interpreter/
  llm/
  utils/
  workflows/
```

## Component summaries

*(high-level modules)*
- `src/__main__.py` - main CLI entrypoint. Loads config, resolves defaults, dispatches subcommands. Reuses functions from `src/<module>/__main__.py`. Mainly calls functions from src/workflows or low-level modules directly.
- `src/workflows/` - contains high-level **user-facing** workflows, reusing lower-level modules.
  Place where you create your custom workflows for your applications.
- `src/config/` - contains tools to work with application config.

*(low-level modules, must not be dependable on high-level modules,
minimal dependencies between each other are allowed)*
- `src/llm/` - contains all functions to interract with language models
  (OpenAI-compatible client wrapper).
- `src/db/` - contains tools to work on document database (store, retrieve
  documents, create indexes)
- `src/utils/` - various utils for 

## Runtime flow (high level)

1. User runs `shai ...`, which is processed by `src/__main__.py`.
2. `src/__main__.py` loads configurations, then calls handlers either from `src/workflows` or from low-level modules directly. (`src/workflows` may reuse functions from low-level modules)

## Testing

Tests are put into `test/` folder. Main entrypoint is `test/test.py`. Read `test/README.md`.
