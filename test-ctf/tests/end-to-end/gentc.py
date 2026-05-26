#!/usr/bin/env python3
"""
This file creates "summarize-m" and "summarize-l" for test_cases.json from
"queries" from test_cases.plain.json (copies queries also)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from shAI_ostap4ello.src import db
from shAI_ostap4ello.src.__main__ import init_shai
from shAI_ostap4ello.src.config import get_config_value, load_config
from shAI_ostap4ello.src.workflows.rag import load_related_documents

def _load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    test_cases = _load_test_cases(Path("./test_cases.plain.json"))
    assert type(test_cases) is dict
    cnt = len(test_cases["queries"])

    # Setup client
    load_config()
    init_shai()
    api_key = get_config_value("llm", "api_key", str)
    api_base_url = get_config_value("llm", "api_base_url", str)
    db_path = Path("../sample-100/docs/")
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    embed_model = get_config_value("llm", "embed_model", str)
    passage_size = 20

    try:
        db.build(
            db_path=str(db_path),
            client=client,
            model=embed_model,
            section_rows=passage_size
        )
    except Exception as e:
        print(f"Index creation warning: {e}")
        raise SystemExit(1)

    test_cases["summarize-m"] = []
    for i in range(cnt):
        try:
            results = db.search(
                db_path=str(db_path),
                client=client,
                query=test_cases["queries"][i],
                top_k=5,
            )
            contents = load_related_documents(results)
            text = "\n\n".join(["%s\n%s" % (p, c) for _, p, c in contents])
            test_cases["summarize-m"].append(text)

        except Exception as e:
            print(f"Test case {i} error: {e}")

    try:
        db.build(
            db_path=str(db_path),
            client=client,
            model=embed_model,
            section_rows=0
        )
    except Exception as e:
        print(f"Index creation warning: {e}")
        raise SystemExit(1)

    test_cases["summarize-l"] = []
    for i in range(cnt):
        try:
            results = db.search(
                db_path=str(db_path),
                client=client,
                query=test_cases["queries"][i]["query"],
                top_k=5,
            )
            contents = load_related_documents(results)
            text = "\n\n".join(["%s\n%s" % (p, c) for _, p, c in contents])
            test_cases["summarize-l"].append(text)

        except Exception as e:
            print(f"Test case {i} error: {e}")

    f = open("./test_cases.json", "w")
    f.write(json.dumps(test_cases))
    f.close()

main()
