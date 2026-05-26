# test_cases.json Structure & Format

## Overview

The `test_cases.json` file defines a **hierarchical tree of CLI command invocations** for regression testing. The test runner (`test_integration.py`) recursively traverses this tree and executes each command, counting passes and failures.

## JSON Schema

Each test case is one of two types:

### 1. **Array Test Case** (Leaf Command)

A simple list of strings representing command-line arguments:

```json
["subcommand", "arg1", "arg2"]
```

**Behavior:**
- Appended to the base argv accumulated from parent nodes
- Example: `base_argv=["shai", "llm"]` + `["generate", "hello"]` → executes `shai llm generate hello`
- Execution: Runs the command and counts as one test case (PASS if exit code 0, FAIL otherwise)

### 2. **Object Test Case** (Branch Command)

An object with two required keys:

```json
{
  "argv": ["subcommand"],
  "children": [
    ["--help"],
    ["-h"],
    { "argv": ["nested"], "children": [...] }
  ]
}
```

**Keys:**
- **`argv`** (required, list of strings): Arguments to append to base argv
- **`children`** (required, list): Child test cases (arrays or objects)

**Behavior:**
- Accumulates base argv with its own `argv`
- Recursively processes all children with the new accumulated base argv
- Does not generate a test case itself; only serves as a branch point

## Execution Model

The test runner uses recursive depth-first traversal:

```python
def _recursive_run(test_case: dict | list | str, base_argv: list):
    if isinstance(test_case, list):
        # Leaf: append to base and execute
        argv = base_argv + test_case
        shai(argv)  # count as 1 test
    elif isinstance(test_case, dict):
        # Branch: accumulate argv, recurse into children
        argv = base_argv + test_case["argv"]
        for child in test_case["children"]:
            _recursive_run(child, argv)
```

### Concrete Example

```json
{
  "argv": ["shai"],
  "children": [
    ["-h"],
    {
      "argv": ["llm"],
      "children": [
        ["-h"],
        ["generate", "hello"]
      ]
    }
  ]
}
```

**Test cases executed:**
1. `shai -h` (root child, array)
2. `shai llm -h` (llm.child[0], array)
3. `shai llm generate hello` (llm.child[1], array)

**Result: 3 test cases**

## Current Implementation (test_cases.json)

### Command Tree

```
shai (root)
├── -h, --help                    [direct tests]
├── db
│   ├── -h, --help
│   ├── check                     [direct test]
│   ├── build
│   │   └── -h, --help
│   └── search
│       └── -h, --help
├── llm
│   ├── -h, --help
│   ├── generate -h              [direct test]
│   └── embed -h                 [direct test]
├── utils
│   ├── -h, --help
│   ├── is_ollama_running -h     [direct test]
│   ├── ls_models -h             [direct test]
│   ├── pull_model -h            [direct test]
│   ├── rm_model -h              [direct test]
│   ├── start_ollama -h          [direct test]
│   ├── stop_ollama -h           [direct test]
│   ├── convert_man_pages -h     [direct test]
│   └── fetch_man_db -h          [direct test]
├── workflows
│   ├── -h, --help
│   └── find -h                  [direct test]
├── init
│   └── -h, --help
├── interpreter
│   └── -h, --help
└── --create-config -h           [direct test]
```

### Coverage Summary

- **Total test cases:** 31
- **Top-level commands tested:** 6 (db, llm, utils, workflows, init, interpreter)
- **Utility commands tested:** 8
- **Test strategy:** Help text validation (quick regression detection)

## Test Strategy & Design

### Why This Approach?

1. **Fast Regression Detection:** Help text tests don't require Ollama or Docker
2. **Comprehensive Coverage:** Tests all CLI entry points and subcommands
3. **Shallow Args:** Minimal arguments avoid side effects and long runtime
4. **Parser Validation:** `-h` tests verify CLI argument parsing is intact

### What's NOT Tested

- Full workflows (no actual LLM calls, DB building, etc.) — these are separate test suites
- Long-running operations (e.g., `fetch_man_db`, `db build` without a real DB)
- Error handling edge cases — limited to help text parsing

### Adding New Tests

To add a new command to the test suite:

1. **Identify the command path:** `shai module subcommand`
2. **Create/update the parent object:** Add `{ "argv": ["module"], "children": [...] }`
3. **Add the test:** Append `["subcommand", "-h"]` to children (or relevant args)
4. **Verify:** Run `python3 test_integration.py` with updated `test_cases.json`

Example:

```json
{
  "argv": ["newmodule"],
  "children": [
    ["-h"],
    ["newsubcommand", "-h"]
  ]
}
```

## Integration with test_integration.py

The test runner (`test_integration.py`):

1. Loads test cases from `test_cases.json`
2. Calls `_recursive_run(root_test_case, [])`
3. For each array test case, executes `shai(argv)` and counts result
4. Outputs `score/total passed` and writes results to JSON

## Running Tests

```bash
cd test-ctf/tests
python3 test_integration.py --config tests.config.json
```

Results are written to `results_file` specified in config, or to console.
