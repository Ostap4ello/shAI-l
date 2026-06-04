# CTF - Config-driven Test Framework (AKA Cool Test Framework)

## Overview
This framework provides a structured way to execute tests as separate independent modules with their own configuration. Tests are registered in a central INI config file (`config.conf`), and test execution is defined in separate INI files, passed to the wrapper (`test.py`). This allows for flexible test management, easy addition of new tests, and clear separation of test logic and configuration. The framework handles loading test modules, parsing configs, executing tests, and collecting summaries.

## Architecture

### Main Components

- **config.conf** - Test registry defining available tests, maps test name to implementation file (e.g., `[test1]` → `./tests/test1.py`)

- **test.py** - Test runner
  - Loads test definitions from `config.conf`
  - Parses run config file (e.g., `test_config_1.conf`) to get configs for
    executing tests
  - Executes tests and prints summaries

- User-defined test modules that implement specific tests.

### Test Module Structure

Each test module must have main file that contains:
- `TEST_CONFIG_SCHEMA` - Dict with config keys and default values
- `TEST_NAME`, `TEST_DESCRIPTION` - Human-readable description and name for the
  test
- `run_test(config: dict) -> str` - Callback function used by wrapper to execute
  the test and to which the parsed config from the run config file is passed.
  Returns a string summary of the test results to be shown in the final output.

## Configuration

### Test Definition (config.conf)
Defines which tests are available and where they are implemented. Sections are
named `<test_name>`. Example:
```ini
[testexample1]
test_file = examplevalue
```

**NOTE**: <test_name> must not contain dots (`.`).

Section name becomes the test identifier used in run configs. When a new test is added, add a new section here.

### Test Run Config (e.g., test_config_1.conf)

Defines which tests to run and with what parameters. Sections are named `<test_name>.<id>. Example:
```ini
[testexample1.run1]
exampleconfig_field = 
```
Then, the testexample1 test will be provided with dict `{"exampleconfig_field":
value}` when executed. The test can have multiple sections (e.g.,
`testexample1.run2`) to run the same test with different configs.


## Usage

### Create default run config

```bash
python3 test.py
```
Shows help message with usage instructions.

```bash
python3 test.py create
```

Creates default config.conf and run_config.conf.


### Run tests

```bash
python3 test.py run ./cfg/test_config_1.conf
```

1. Loads test definitions from `config.conf`
2. Loads run config from provided path
3. Extracts `config_file` field from first run config section to load shAI app config
4. Executes all matching test sections, prints summaries


## Extending

To add a new test:
1. Create `tests/testnew.py` with `TEST_NAME="test"`, `TEST_CONFIG_SCHEMA`, and `run_test()`
2. Add new section to `config.conf`:
   ```ini
   [test3]
   test_file = ./tests/testnew.py
   ```
3. Create new section in run config for the test:
   ```ini
   [test3.run1]
   config_field = value
   ```
*NOTE: No changes to test.py needed - config-driven discovery handles it automatically.*
