#!/usr/bin/env python3
"""End-to-end latency testing"""

import signal
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List

from shAI_ostap4ello.src.__main__ import main as _shai

import logging

logger = logging.getLogger(__name__)

TEST_NAME = "test6"
TEST_DESCRIPTION = "End-to-end application latency testing for main functionality"
TEST_CONFIG_SCHEMA = {
    "test_cases_file": "",
    "results_file": "",
    "config_file": "",
    "section_size": "",
}


def _load_test_cases(path: Path) -> List[Dict[str, dict]]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return json.load(f)

def shai(argv_base: list, argv_mod: list):
    argv = argv_base + argv_mod
    try:
        _shai(argv)
    except SystemExit as e:
        if e.code != 0:
            logger.error(f"shai with argv {argv} exited with code {e.code}")
    except Exception as e:
        logger.error(f"shai with argv {argv} exited unhandled Exception {e}")

def run_test(config: Dict[str, Any]) -> str:
    """Run End-to-end application latency testing"""

    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    test_cases_file = Path(config["test_cases_file"])
    results_file = config.get("results_file")
    shai_config = str(config.get("config_file"))
    section_size = int(config.get("section_size", 0))
    extended_search = bool(config.get("extended_search", False))
    tc = _load_test_cases(test_cases_file)
    assert type(tc) is dict

    argv_base = ["shai", "--config", shai_config]

    repeats = 10
    results = {}

    # TODO: quiet mode for logging

    # Building index
    argv_base_db = argv_base + ["db", "--log-level", "ERROR", "build", "--section-rows", f"{section_size}"]

    logger.info("Starting index build latency tests on sample-10")
    lat = 0
    for _ in range(repeats):
        tc_start = time.time()
        shai(argv_base_db, ["--db-path", "./tests/sample-10/docs/"])
        lat += (time.time() - tc_start)
    results["build-10"] = lat / repeats

    logger.info("Starting index build latency tests on sample-100")
    lat = 0
    for _ in range(repeats):
        tc_start = time.time()
        shai(argv_base_db, ["--db-path", "./tests/sample-100/docs/"])
        lat += (time.time() - tc_start)
    results["build-100"] = lat / repeats

    logger.info("Starting index build latency tests on end-to-end (1000)")
    lat = 0
    for _ in range(repeats):
        tc_start = time.time()
        shai(argv_base_db, ["--db-path", "./tests/end-to-end/docs/"])
        lat += (time.time() - tc_start)
    results["build-1000"] = lat / repeats

    # Searching
    argv_base_dbs = argv_base + ["db", "--log-level", "ERROR", "search"]
    if extended_search:
        argv_base_dbs.append("--extended")
    tcs = tc["queries"]
    assert type(tcs) is list

    logger.info("Starting plain search latency tests on sample-10")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_dbs, [f"{tcs[i]}", "--db-path", "./tests/sample-10/docs/"])
        lat += (time.time() - tc_start)
    results["search-10"] = lat / repeats

    logger.info("Starting plain search latency tests on sample-100")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_dbs, [f"{tcs[i]}", "--db-path", "./tests/sample-100/docs/"])
        lat += (time.time() - tc_start)
    results["search-100"] = lat / repeats

    logger.info("Starting plain search latency tests on end-to-end (1000)")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_dbs, [f"{tcs[i]}", "--db-path", "./tests/end-to-end/docs/"])
        lat += (time.time() - tc_start)
    results["search-1000"] = lat / repeats

    # Plain generation
    argv_base_gen = argv_base + ["llm", "--log-level", "ERROR", "generate"]
    tcgs = tc["queries"]      # Example short inputs (simulate question answering on short query level)
    tcgm = tc["summarize-m"]  # Example med inputs (simulate summarization on paragraph level)
    tcgl = tc["summarize-l"]  # Example long inputs (simulate summarization on doc level)
    assert type(tcgs) is list
    assert type(tcgm) is list
    assert type(tcgl) is list

    logger.info("Starting plain generation latency tests on short queries")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_gen, [f"{tcgs[i]}"])
        lat += (time.time() - tc_start)
    results["generate-s"] = lat / repeats

    logger.info("Starting plain generation latency tests on medium inputs")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_gen, [f"{tcgm[i]}"])
        lat += (time.time() - tc_start)
    results["generate-m"] = lat / repeats

    logger.info("Starting plain generation latency tests on long inputs")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_gen, [f"{tcgl[i]}"])
        lat += (time.time() - tc_start)
    results["generate-l"] = lat / repeats


    # RAG wf
    argv_base_wf = argv_base + ["workflows", "--log-level", "ERROR", "find"]
    if extended_search:
        argv_base_wf.append("--extended-search")
    tcs = tc["queries"]
    assert type(tcs) is list

    logger.info("Starting RAG workflow latency tests on sample-10")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_wf, [f"{tcs[i]}", "--db-path", "./tests/sample-10/docs/"])
        lat += (time.time() - tc_start)
    results["rag-10"] = lat / repeats

    logger.info("Starting RAG workflow latency tests on sample-100")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_wf, [f"{tcs[i]}", "--db-path", "./tests/sample-100/docs/"])
        lat += (time.time() - tc_start)
    results["rag-100"] = lat / repeats

    logger.info("Starting RAG workflow latency tests on end-to-end (1000)")
    lat = 0
    for i in range(repeats):
        tc_start = time.time()
        shai(argv_base_wf, [f"{tcs[i]}", "--db-path", "./tests/end-to-end/docs/"])
        lat += (time.time() - tc_start)
    results["rag-1000"] = lat / repeats

    # Save results
    output = {
        "test": TEST_NAME,
        "description": TEST_DESCRIPTION,
        "results": results,
    }

    if results_file:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    summary = (
        f"{TEST_NAME}: {TEST_DESCRIPTION}"
        f"\nResults:\n" + "\n".join(f"{k}: {v:.2f} seconds" for k, v in results.items())
    )
    return summary
