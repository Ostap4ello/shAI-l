#!/usr/bin/env python3
"""LLM classification test: natural language vs bash script."""

import time
import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from shAI_ostap4ello.src.config import get_config_value, load_config

import logging

from shAI_ostap4ello.src.rag.classifier import classify_is_bash

logger = logging.getLogger(__name__)

TEST_NAME = "test3"
TEST_DESCRIPTION = (
    "LLM classification test: natural language (natlang) vs bash script (bash)"
)
TEST_CONFIG_SCHEMA = {
    "test_cases_file": "",
    "results_file": "",
    "config_file": "",
}

_prompt = f"""Classify the following text as either "natural_language" or "bash_script".
Respond with ONLY the word: natural_language or bash_script

Text:
%s

Classification:"""


def _load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return json.load(f)


def _classify_text(client: OpenAI, model: str, text: str) -> str:
    """
    Classify text as natural_language or bash_script using LLM.

    Returns classification as string: "natural_language" or "bash_script"
    """
    prompt = format(_prompt % text)
    try:
        if classify_is_bash(client, model, text):
            return "bash"
        else:
            return "natlang"
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return "error"


def run_test(config: Dict[str, Any]) -> str:
    """Run LLM classification test."""
    test_cases_file = Path(config["test_cases_file"])
    results_file = config.get("results_file")
    shai_config = str(config.get("config_file"))

    test_cases = _load_test_cases(test_cases_file)

    # Setup client
    load_config(shai_config)
    api_key = get_config_value("llm", "api_key", str)
    api_base_url = get_config_value("llm", "api_base_url", str)
    model = get_config_value("llm", "model", str)
    client = OpenAI(api_key=api_key, base_url=api_base_url)

    # Run classifications
    correct = 0
    latencies = []
    results_list = []

    for i, tc in enumerate(test_cases, 1):
        tc_start = time.time()
        expected_label = tc["label"]

        try:
            predicted_label = _classify_text(client, model, tc["query"])
            is_correct = predicted_label == expected_label
            if is_correct:
                correct += 1
        except Exception as e:
            logger.error(f"Test case {i} error: {e}")
            predicted_label = "error"
            is_correct = False

        tc_latency = time.time() - tc_start
        latencies.append(tc_latency)
        results_list.append(
            {
                "id": tc.get("id", i),
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": 1 if is_correct else 0,
                "latency": round(tc_latency, 4),
            }
        )

    # Calculate metrics
    n = len(test_cases)
    accuracy = (correct / n * 100) if n > 0 else 0.0

    # Save results
    output = {
        "test": TEST_NAME,
        "correct/total": f"{correct}/{n}",
        "accuracy": f"{accuracy:.1f}%",
        "model": model,
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
        f"{TEST_NAME}: {TEST_DESCRIPTION}"
        f"accuracy={output['accuracy']} ({output['correct']}/{output['total']}) | "
        f"avg_latency={output['avg_latency']}s"
    )
    return summary
