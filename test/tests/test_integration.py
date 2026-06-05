#!/usr/bin/env python3
"""
Integration testing (via cli) for main shAI functionality, with a focus on
regression testing
"""

import os
import re
import signal
import subprocess
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

from shAI_ostap4ello.src.__main__ import main as _shai

import logging

logger = logging.getLogger(__name__)

TEST_NAME = "test_integration"
TEST_DESCRIPTION = "Integration testing (via cli) for main shAI functionality, with a focus on regression testing"
TEST_CONFIG_SCHEMA = {
    "test_cases_file": "",
    "results_file": "",
    "config_file": "",
    "db_path": "",
}


def _load_test_cases(path: Path) -> List[Dict[str, dict]]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return json.load(f)


def shai(argv):
    global score, total, err
    total += 1

    result = subprocess.run(
        argv, capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.info(f"FAIL: argv {argv} exited with code {result.returncode}")
        for i in range(len(argv)):
            argv[i] = re.sub(" ", "\\ ", argv[i])
        err.append(" ".join(argv))
        logger.debug(f"stderr: {result.stderr}\n")
    else:
        logger.info(f"PASS: argv {argv} completed successfully")
        logger.debug(f"stdout: {result.stdout}\n")
        score += 1


def _recursive_run(test_case: dict | list | str, base_argv: list):
    if type(test_case) is list:
        for i in range(len(test_case)):
            if type(test_case[i]) is not str:
                logger.error(f"Test case {test_case[i]} in list is not a string")
                return
            if test_case[i] == "$DB_PATH":
                test_case[i] = db_path
        argv = base_argv + test_case
        shai(argv)
    elif type(test_case) is dict:
        if not "argv" in test_case or not "children" in test_case:
            logger.error(f"Test case {test_case} is missing 'argv' or 'children' key")
            return 3
        argv = base_argv + test_case["argv"]
        children = test_case.get("children", [])
        if len(children) == 0:
            logger.warning(f"Empty test case list encountered in {test_case}")
        for c in children:
            _recursive_run(c, argv)
    else:
        logger.error(f"Test case {test_case} is of unsupported type {type(test_case)}")


def run_test(config: Dict[str, Any]) -> str:
    """
    Run Integration testing for main shAI functionality, with a focus on
    regression testing
    """

    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    test_cases_file = Path(config["test_cases_file"])
    results_file = config.get("results_file")
    global db_path
    db_path = config.get("db_path", "./tests/regress/docs/")

    tc = _load_test_cases(test_cases_file)
    assert type(tc) is list

    # TODO: quiet mode for logging

    # Building index

    global score, total, err
    score = 0
    total = 0
    err = []
    for t in tc:
        _recursive_run(t, [])

    # Save results
    output = {
        "test": TEST_NAME,
        "description": TEST_DESCRIPTION,
        "results": f"{score}/{total} test cases passed",
    }
    if len(err) > 0:
        # TODO: fix warning wigh assert
        output["failed"] = err
        logger.info("List of failed commands:")
        for e in err:
            logger.info(f"-> {e}")

    if results_file:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    summary = f"{TEST_NAME}: {TEST_DESCRIPTION} | " f"results={output['results']}"

    return summary
