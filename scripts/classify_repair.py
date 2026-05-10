"""
Classify LLM repair behavior for every follow-up turn (turn_id >= 2).

For each follow-up exchange:
  - repair_outcome : one of {Successful Repair, Partial Repair,
                              Failed Repair, Repeated Misconception}
  - repair_behaviors: comma-separated subset of six tags

Writes results to data/repair_labels.csv.

Usage:
  python scripts/classify_repair.py             # classify all unscored
  python scripts/classify_repair.py --dry-run   # print prompts, no API calls
  python scripts/classify_repair.py --force     # re-classify everything
  python scripts/classify_repair.py --limit 10  # first 10 follow-up turns

Requires: OPENAI_API_KEY env var.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found. Run: pip install openai")

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV   = os.path.join(BASE, "data", "processed.csv")
OUT_CSV  = os.path.join(BASE, "data", "repair_labels.csv")

REPAIR_OUTCOMES = [
    "Successful Repair",
    "Partial Repair",
    "Failed Repair",
    "Repeated Misconception",
]

REPAIR_BEHAVIORS = [
    "Changed strategy after follow-up",
    "Used simplification",
    "Used step-by-step structure",
    "Used analogy or visual explanation",
    "Used concrete example",
    "Student showed further confusion afterward",
]

SYSTEM_PROMPT = f"""You are an expert educational researcher analysing student–LLM tutoring conversations.

You will be shown a tutoring exchange consisting of:
  1. The LLM's previous response (what the LLM said before the student's follow-up)
  2. The student's follow-up message (showing possible confusion or a new question)
  3. The LLM's repair response (how the LLM responded to the follow-up)

Your task is to classify this exchange on two dimensions:

--- REPAIR OUTCOME (choose exactly one) ---
  "Successful Repair"       – The LLM's repair response clearly resolves the student's confusion or question.
  "Partial Repair"          – The LLM's repair response addresses the confusion partially but leaves gaps.
  "Failed Repair"           – The LLM repeats roughly the same explanation without meaningfully helping.
  "Repeated Misconception"  – The LLM reinforces or introduces an error in its repair response.

--- REPAIR BEHAVIORS (choose all that apply; may be empty list) ---
  "Changed strategy after follow-up"        – The LLM noticeably shifted its explanatory approach.
  "Used simplification"                     – The LLM simplified language or reduced complexity.
  "Used step-by-step structure"             – The LLM broke the explanation into numbered steps.
  "Used analogy or visual explanation"      – The LLM used an analogy, metaphor, or described a visual.
  "Used concrete example"                   – The LLM provided a specific worked example or scenario.
  "Student showed further confusion afterward" – The student's follow-up text signals remaining confusion
                                                (e.g., "I still don't understand", "wait but…", restating
                                                the same question, or introducing a new misconception).

Respond with ONLY a JSON object with exactly these two keys:
{{
  "repair_outcome": "<one of the four outcomes>",
  "repair_behaviors": ["<behavior>", ...]
}}

Output ONLY the JSON object, no prose.
"""


def build_prompt(prev_llm: str, student_followup: str, repair_llm: str) -> str:
    def clip(text, n=500):
        return (text[:n] + "…") if len(text) > n else text

    return (
        f"Previous LLM response:\n{clip(prev_llm)}\n\n"
        f"Student follow-up:\n{clip(student_followup)}\n\n"
        f"LLM repair response:\n{clip(repair_llm)}"
    )


def call_gpt(client, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content.strip())


def validate(raw: dict) -> dict:
    outcome = raw.get("repair_outcome", "")
    if outcome not in REPAIR_OUTCOMES:
        outcome = "Partial Repair"

    behaviors = raw.get("repair_behaviors", [])
    if not isinstance(behaviors, list):
        behaviors = []
    behaviors = [b for b in behaviors if b in REPAIR_BEHAVIORS]

    return {"repair_outcome": outcome, "repair_behaviors": behaviors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key) if not args.dry_run else None

    # Load processed.csv and index by (file, turn_id)
    with open(IN_CSV, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    turns_index: dict[tuple, dict] = {}
    for r in all_rows:
        turns_index[(r["file"], r["turn_id"])] = r

    # Load existing repair labels to support incremental runs
    existing: dict[tuple, dict] = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[(r["file"], r["turn_id"])] = r

    # Identify follow-up turns (turn_id >= 2) and sort them
    followups = [r for r in all_rows if int(r["turn_id"]) >= 2]
    followups.sort(key=lambda r: (r["file"], int(r["turn_id"])))

    results = []
    scored = skipped = 0

    for row in followups:
        if args.limit and scored >= args.limit:
            break

        key = (row["file"], row["turn_id"])
        if key in existing and not args.force:
            results.append(existing[key])
            skipped += 1
            continue

        prev_turn_id = str(int(row["turn_id"]) - 1)
        prev_row = turns_index.get((row["file"], prev_turn_id))
        if not prev_row:
            skipped += 1
            continue

        prev_llm      = prev_row.get("llm_text", "").strip()
        student_fup   = row.get("student_text", "").strip()
        repair_llm    = row.get("llm_text", "").strip()

        if not prev_llm or not student_fup or not repair_llm:
            skipped += 1
            continue

        prompt = build_prompt(prev_llm, student_fup, repair_llm)

        if args.dry_run:
            print(f"\n--- {row['file']} turn {row['turn_id']} ---")
            print(prompt[:400])
            scored += 1
            continue

        try:
            raw = call_gpt(client, prompt, args.model)
            labels = validate(raw)
        except Exception as e:
            print(f"  [ERROR] {row['file']} t{row['turn_id']}: {e}", file=sys.stderr)
            scored += 1
            continue

        result = {
            "file":             row["file"],
            "conv_id":          row["conv_id"],
            "turn_id":          row["turn_id"],
            "label_source":     row.get("label_source", ""),
            "repair_outcome":   labels["repair_outcome"],
            "repair_behaviors": "; ".join(labels["repair_behaviors"]),
        }
        results.append(result)

        print(f"[{scored+1}] {row['file']} t{row['turn_id']}: "
              f"{labels['repair_outcome']} | {labels['repair_behaviors']}")

        scored += 1
        time.sleep(0.2)

    if args.dry_run:
        print(f"\n[dry-run] Would classify {scored} follow-up turns.")
        return

    # Merge skipped (existing) rows that weren't re-processed
    all_results = {(r["file"], r["turn_id"]): r for r in results}
    for key, r in existing.items():
        if key not in all_results:
            all_results[key] = r

    fieldnames = ["file", "conv_id", "turn_id", "label_source",
                  "repair_outcome", "repair_behaviors"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(all_results.values(),
                                key=lambda r: (r["file"], int(r["turn_id"]))))

    print(f"\nDone. Classified: {scored}, skipped (already done / missing text): {skipped}.")
    print(f"Written: {OUT_CSV}")


if __name__ == "__main__":
    main()
