#!/usr/bin/env python3
"""End-to-end latency testing"""

import signal
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)

TEST_NAME = "test6"
TEST_DESCRIPTION = "End-to-end application latency testing for main functionality"
TEST_CONFIG_SCHEMA = {
    "test_cases_file": "",
    "results_file": "",
    "config_file": "",
    "db_path": "",
    "section_rows": "",
    "extended_search": "",
}


def _load_test_cases(path: Path) -> List[Dict[str, dict]]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return json.load(f)


def shai(argv_base: list, argv_mod: list) -> subprocess.CompletedProcess:
    argv = argv_base + argv_mod
    result = subprocess.run(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        logger.error(f"shai with argv {argv} exited with code {result.returncode}")
        logger.error(f"stderr: {result.stderr}\n")
    else:
        logger.debug(f"shai with argv {argv} completed successfully")
        logger.debug(f"stdout: {result.stdout}\n")
    return result


def warmup_embedding_model(argv_base):
    shai(argv_base, ["llm", "embed", "say one word"])


def warmup_generation_model(argv_base):
    shai(argv_base, ["llm", "generate", "say one word"])


def run_test(config: Dict[str, Any]) -> str:
    """Run End-to-end application latency testing"""

    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted test6. Exiting cleanly.", file=sys.stderr)
        shai(argv_base, ["utils", "stop_ollama", "--name", "test6-ollama", "--remove"])
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    test_cases_file = Path(config["test_cases_file"])
    results_file = config.get("results_file")
    shai_config = str(config.get("config_file"))
    db_path = str(config.get("db_path", "./tests/sample-10/docs/"))
    db_label = Path(db_path).parent.name
    section_rows = int(config.get("section_rows", 0))
    extended_search = config.get("extended_search", "false").lower() == "true"
    repeats = int(config.get("repeats", 1))

    testcases = _load_test_cases(test_cases_file)
    assert type(testcases) is dict

    argv_base = ["shai", "--config", shai_config]

    results = {}

    logger.info("Starting Ollama for test6...")
    result = shai(
        argv_base, ["utils", "start_ollama", "--create", "--name", "test6-ollama", "--gpus", "none"]
    )
    if result.returncode != 0:
        logger.error("Failed to start Ollama for test6. Exiting.")
        return f"{TEST_NAME}: Failed to start Ollama. See logs for details."

    logger.info("Warming up embedding model...")
    tc_start = time.time()
    warmup_embedding_model(argv_base)
    results["embed-model-warmup"] = time.time() - tc_start

    logger.info(f"Starting index build latency tests on {db_label}")
    argv_base_db = argv_base + ["db", "build", "--section-rows", f"{section_rows}"]

    lat = 0
    for _ in range(repeats):
        tc_start = time.time()
        shai(argv_base_db, ["--db-path", db_path])
        lat += time.time() - tc_start
    results[f"build-{db_label}"] = lat / repeats

    logger.info(f"Starting search latency tests on {db_label}")
    argv_base_search = argv_base + ["db", "search"]
    if extended_search:
        argv_base_search.append("--extended-search")

    queries = testcases["queries"]
    assert type(queries) is list

    lat = 0
    for q in queries:
        tc_start = time.time()
        shai(argv_base_search, [f"{q}", "--db-path", db_path])
        lat += time.time() - tc_start
    results[f"search-{db_label}"] = lat / len(queries)

    logger.info("Restarting Ollama to clear model cache...")
    shai(argv_base, ["utils", "stop_ollama", "--name", "test6-ollama", "--remove"])
    shai(argv_base, ["utils", "start_ollama", "--name", "test6-ollama", "--create", "--gpus", "none"])

    logger.info("Warming up embedding model...")
    warmup_embedding_model(argv_base)

    logger.info("Warming up generation model...")
    tc_start = time.time()
    warmup_generation_model(argv_base)
    results["generation-model-warmup"] = time.time() - tc_start

    logger.info(f"Starting RAG workflow latency tests on {db_label}")
    argv_base_wf = argv_base + ["workflows", "find"]
    if extended_search:
        argv_base_wf.append("--extended-search")

    queries = testcases["queries"]
    assert type(queries) is list

    lat = 0
    for q in queries:
        tc_start = time.time()
        shai(argv_base_wf, [f"{q}", "--db-path", db_path])
        lat += time.time() - tc_start
    results[f"rag-{db_label}"] = lat / len(queries)

    # # Plain generation tests
    # argv_base_gen = argv_base + ["llm", "generate"]
    #
    # tcgs = tc["queries"]
    # assert type(tcgs) is list
    #
    # logger.info(f"Starting plain generation latency tests on {db_label}")
    # lat = 0
    # for i in range(tcgs):
    #     tc_start = time.time()
    #     shai(argv_base_gen, [f"{tcgs[i]}"])
    #     lat += time.time() - tc_start
    # results[f"generate-{db_label}"] = lat / repeats

    # Bringing down Ollama
    logger.info("Stopping Ollama for test6...")
    result = shai(
        argv_base, ["utils", "stop_ollama", "--name", "test6-ollama", "--remove"]
    )
    if result.returncode != 0:
        logger.error(
            "Failed to stop Ollama for test6. Please check Ollama status manually."
        )
    else:
        logger.info("Ollama stopped successfully for test6.")

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

    summary = f"{TEST_NAME}: {TEST_DESCRIPTION}" f"\nResults:\n" + "\n".join(
        f"{k}: {v:.2f} seconds" for k, v in results.items()
    )
    return summary
