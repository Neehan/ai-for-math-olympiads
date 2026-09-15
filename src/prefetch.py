"""Fetch protected experiment inputs before Docker closes network access."""

import json
import os
import sys
import urllib.request
from pathlib import Path

from src.constants import FETCH_TIMEOUT_SECONDS, SELECTION_URL, data_source_urls


def prefetch(stage: str, arm: str | None, directory: Path) -> None:
    """Write disposable loader inputs, never full solutions for generation."""
    if stage not in {"run", "audit", "state-audit"}:
        raise ValueError(f"Unknown stage: {stage}")
    urls = data_source_urls(arm)
    directory.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict]] = {}

    def fetch(url: str) -> list[dict]:
        if url not in cache:
            with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:
                cache[url] = [
                    json.loads(line)
                    for line in response.read().decode("utf-8").splitlines()
                    if line.strip()
                ]
        return cache[url]

    def write(name: str, rows: list[dict]) -> None:
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    # alt-hint uses its solutions file as the statement source. Strip proofs
    # here, before anything is written to the generation container filesystem.
    problems = [
        {
            key: record[key]
            for key in ("problem_id", "statement", "domain")
            if key in record
        }
        for record in fetch(urls["problems"])
    ]
    write("problems", problems)
    write("hints", fetch(urls["hints"]))
    write("outlines", fetch(urls["outlines"]))
    print("datasets prefetched")
    if arm in {"selection", "selection-no-problem"}:
        write("selection", fetch(SELECTION_URL))
        print("selection candidates prefetched")
    if stage in {"audit", "state-audit"}:
        write("solutions", fetch(urls["solutions"]))
        print("reference solutions prefetched")


if __name__ == "__main__":
    prefetch(sys.argv[1], os.environ.get("HARNESS_ARM"), Path("/run/contest"))
