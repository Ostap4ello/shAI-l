#!/usr/bin/env python3
"""RAG precision@k test on passage scope using extended search."""

import time
import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from shAI_ostap4ello.src.config import get_config_value, load_config
from shAI_ostap4ello.src.db.db import build, search, search_in_files_dynamic

import logging

logger = logging.getLogger(__name__)

TEST_NAME = "test2"
TEST_DESCRIPTION = "DB is-in-top-k and MRR test on passage scope using extended search"
TEST_CONFIG_SCHEMA = {
    "collection_dir": "",
    "test_cases_file": "",
    "batch_size": "32",
    "top_k": "10",
    "top_k_extended": "10",
    "index_dir_name": ".index",
    "results_file": "",
    "config_file": "",
}


def _load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return json.load(f)


def _check_section_match(
    result_path: str,
    result_begin_line: int,
    result_end_line: int,
    expected_filename: str,
    expected_line: int,
) -> bool:
    """Check if result matches expected document."""
    logger.debug(
        f"Comparing result section '{result_path}:{result_begin_line}-{result_end_line}'"
        + f" with expected '{expected_filename}:{expected_line}'"
    )
    result_name = Path(result_path).name
    expected_name = Path(expected_filename).name
    return (
        result_name == expected_name
        and result_begin_line <= expected_line <= result_end_line
    )


def run_test(config: Dict[str, Any]) -> str:
    """Run test, save results, return summary."""
    collection_dir = Path(config["collection_dir"])
    test_cases_file = Path(config["test_cases_file"])
    batch_size = int(config["batch_size"])
    top_k = int(config["top_k"])
    top_k_extended = int(config["top_k_extended"])
    index_dir_name = config["index_dir_name"]
    results_file = config.get("results_file")
    shai_config = str(config.get("config_file"))

    db_path = collection_dir / "docs"
    test_cases = _load_test_cases(test_cases_file)

    # Client
    load_config(shai_config)
    api_key = get_config_value("llm", "api_key", str)
    api_base_url = get_config_value("llm", "api_base_url", str)
    embed_model = get_config_value("llm", "embed_model", str)
    client = OpenAI(api_key=api_key, base_url=api_base_url)

    # Build index
    logger.info("Building index for passage search...")
    index_start = time.time()
    try:
        build(
            db_path=str(db_path),
            client=client,
            model=embed_model,
            batch_size=batch_size,
            index_path_within_db=index_dir_name,
        )
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")
        return "Index creation failed, test aborted."
    index_latency = time.time() - index_start

    # Run tests with extended top_k for passage search
    top_1_matches = 0
    top_k_matches = 0
    mrr = 0.0
    latencies = []
    results_list = []

    for i, test_case in enumerate(test_cases, 1):
        tc_start = time.time()
        found_rank = 0
        try:
            results = search(
                db_path=str(db_path),
                client=client,
                query=test_case["query"],
                top_k=top_k,
                index_path_within_db=index_dir_name,
            )

            paths = []
            for r in results:
                paths.append(r["metadata"]["path"])

            results = search_in_files_dynamic(
                file_paths=paths,
                client=client,
                model=embed_model,
                query=test_case["query"],
                top_k=top_k_extended,
            )

            for rank, result in enumerate(results, 1):
                if _check_section_match(
                    result.get("metadata", {}).get("path"),
                    int(result.get("metadata", {}).get("from")),
                    int(result.get("metadata", {}).get("to")),
                    test_case["filename"],
                    test_case["line"],
                ):
                    found_rank = rank
                    break

            mrr += 1.0 / found_rank if found_rank > 0 else 0.0
            top_1 = 1 if found_rank == 1 else 0
            top_k_v = 1 if found_rank > 0 else 0
            top_1_matches += top_1
            top_k_matches += top_k_v

        except Exception as e:
            logger.error(f"Test case {i} error: {e}")
            top_1, top_k_v = 0, 0

        tc_latency = time.time() - tc_start
        latencies.append(tc_latency)
        results_list.append(
            {
                "id": test_case.get("id", i),
                "top_1": top_1,
                "top_k": top_k_v,
                "rank": found_rank,
                "latency": round(tc_latency, 4),
            }
        )

    # Save results
    n = len(test_cases)
    output = {
        "test": TEST_NAME,
        "model": embed_model,
        "top_1_matches": f"{top_1_matches}/{n}",
        "top_k_matches": f"{top_k_matches}/{n}",
        "top_1_accuracy": f"{(top_1_matches/n*100):.1f}%" if n > 0 else "0%",
        "top_k_accuracy": f"{(top_k_matches/n*100):.1f}%" if n > 0 else "0%",
        "index_latency": round(index_latency, 4),
        "mrr": round(mrr / n, 4) if n > 0 else 0.0,
        "avg_latency": round(sum(latencies) / len(latencies), 4) if latencies else 0,
        "testcases": results_list,
    }

    if results_file:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    summary = (
        f"{TEST_NAME}: {TEST_DESCRIPTION} | "
        f"top-1={output['top_1_matches']} ({output['top_1_pct']}) | "
        f"top-k={output['top_k_matches']} ({output['top_k_pct']}) | "
        f"MRR={output['mrr']} | "
        f"avg_latency={output['avg_latency']}s | "
        f"index_latency={output['index_latency']}s"
    )
    return summary
