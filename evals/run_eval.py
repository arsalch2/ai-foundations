"""Day 6: your first eval. Runs every test email through the extractor and scores each field.

Usage:  python -m evals.run_eval
Writes evals/results.md so you can paste the table into your README and track changes over time.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.llm import get_client
from app.main import extract_order
from app.schemas import ExtractRequest

DATA = Path(__file__).parent / "order_emails.jsonl"
OUT = Path(__file__).parent / "results.md"
FIELDS = ["is_order", "customer_company", "items", "requested_delivery_date", "language"]


def norm(value):
    return " ".join(str(value).lower().split()) if value is not None else None


def items_key(items) -> set[tuple[str, int]]:
    return {(norm(i["product"]).upper(), int(i["quantity"])) for i in items}


def score(expected: dict, got: dict) -> dict[str, bool]:
    return {
        "is_order": expected["is_order"] == got["is_order"],
        "customer_company": norm(expected["customer_company"]) == norm(got["customer_company"]),
        "items": items_key(expected["items"]) == items_key(got["items"]),
        "requested_delivery_date": expected["requested_delivery_date"] == got["requested_delivery_date"],
        "language": expected["language"] == got["language"],
    }


def main() -> int:
    settings = get_settings()
    client = get_client()
    cases = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows, totals = [], {f: 0 for f in FIELDS}
    started = time.perf_counter()
    for case in cases:
        try:
            res = extract_order(ExtractRequest(email_text=case["email"]), client=client, settings=settings)
            got = res.result.model_dump() if res.result else None
        except Exception as exc:  # count a crash as all-fields-wrong, but keep going
            print(f"  {case['id']}: ERROR {exc}", file=sys.stderr)
            got = None
        checks = score(case["expected"], got) if got else {f: False for f in FIELDS}
        for f, ok in checks.items():
            totals[f] += ok
        rows.append((case["id"], checks))
        print(f"{case['id']:<24} " + " ".join("✓" if checks[f] else "✗" for f in FIELDS))

    n = len(cases)
    seconds = time.perf_counter() - started
    lines = [
        f"## Extraction eval · {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Provider `{settings.llm_provider}`, model `{settings.chat_model}`, {n} emails, {seconds:.1f} s",
        "",
        "| Field | Correct | Accuracy |",
        "| --- | --- | --- |",
        *[f"| {f} | {totals[f]}/{n} | {totals[f] / n:.0%} |" for f in FIELDS],
        "",
        "| Case | " + " | ".join(FIELDS) + " |",
        "| --- | " + " | ".join("---" for _ in FIELDS) + " |",
        *[f"| {cid} | " + " | ".join("✓" if c[f] else "✗" for f in FIELDS) + " |" for cid, c in rows],
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[4:4 + len(FIELDS) + 2]))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
