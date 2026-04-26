#!/usr/bin/env python3
"""Main test runner with config-driven execution."""

from pathlib import Path
from typing import Dict
from os import environ
import importlib.util
import configparser
import sys
import logging

from shAI_ostap4ello.src.config import load_config

logger = logging.getLogger(__name__)

# Test registry: {name: (schema, run_test_func, description)}
_tests: Dict[str, tuple] = {}

DEFAULT_DEBUG_LEVEL = "INFO"


def register_tests(config_path: str) -> None:
    """Load test modules defined in test/config.conf."""
    config = configparser.ConfigParser()
    config.read(Path(config_path).expanduser())

    for section in config.sections():
        if not section.startswith("test"):
            continue

        if section in _tests:
            logger.error(f"Duplicate test name '{section}' in config")
            continue

        if "test_file" not in config[section]:
            logger.error(f"Missing 'test_file' in section {section}")
            continue

        test_file = config[section]["test_file"]
        file_path = (Path(__file__).parent / test_file).resolve()

        if not file_path.exists():
            logger.error(f"Test file not found: {file_path}")
            continue

        spec = importlib.util.spec_from_file_location(section, file_path)
        if not (spec and spec.loader):
            logger.error(f"Failed to load spec for {file_path}")
            continue

        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "TEST_NAME") and hasattr(mod, "run_test"):
                # Register test with name from module
                _tests[section] = (
                    getattr(mod, "TEST_CONFIG_SCHEMA", {}),
                    mod.run_test,
                    getattr(mod, "TEST_DESCRIPTION", ""),
                )
                logger.debug(f"Loaded test: {section} from {test_file}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")


def create_default_configs() -> None:
    # Config
    config = configparser.ConfigParser()
    config["testexample1"] = {"test_file": ""}
    path = Path("./config.conf").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        config.write(f)
    print(f"Config created: {path}")

    # Run config
    config = configparser.ConfigParser()
    config["testexample1.run1"] = {"exampleconfig_field": ""}

    path = Path("./run_config.conf").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        config.write(f)
    print(f"Run config created: {path}")


def run_tests(config_path: str) -> None:
    """Run all configured tests."""
    # Read run config first to get shai config path
    run_config = configparser.ConfigParser()
    run_config.read(Path(config_path).expanduser())

    # Get shai config path from first section (all sections should have same config_file)
    shai_config_path = None
    for section in run_config.sections():
        if "config_file" in run_config[section]:
            shai_config_path = run_config[section]["config_file"]
            break

    if shai_config_path:
        load_config(shai_config_path)

    results = []
    for name, (_, func, desc) in sorted(_tests.items()):
        # Find sections matching the test name in run config
        matching_sections = [
            s for s in run_config.sections() if s.startswith(f"{name}.")
        ]

        if not matching_sections:
            logger.warning(f"No config for test {name}")
            continue

        for section in matching_sections:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Test {name} ({section}): {desc}")
            logger.info(f"{'='*60}")

            cfg_dict = dict(run_config[section])
            try:
                summary = func(cfg_dict)
                results.append((name, section, summary))
                print(f"\n{summary}\n")
            except Exception as e:
                logger.error(f"Test {name} failed: {e}", exc_info=True)
                results.append((name, section, f"ERROR: {e}"))

    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, section, summary in results:
            print(f"{section}: {summary}")
        print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=environ.get("DEBUG_LEVEL", DEFAULT_DEBUG_LEVEL).upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            f"  {sys.argv[0]} create                - Create default run and test configs "
            + "at ./config.conf and ./test_config.conf"
        )
        print(f"  {sys.argv[0]} run <config_path>     - Run tests")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create":
        create_default_configs()
    elif cmd == "run":
        test_def_path = Path(__file__).parent / "config.conf"
        register_tests(str(test_def_path))

        run_config_path = sys.argv[2] if len(sys.argv) > 2 else None
        if not run_config_path:
            logger.error("Error: config_path argument is required")
            sys.exit(1)
        run_tests(run_config_path)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
