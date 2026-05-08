"""
LLM Labeling — adds 6 pedagogical labels to every row in processed.csv.

Labels added per turn:
  satisfaction        : satisfied | neutral | unsatisfied
  confusion           : confused | unclear | not_confused
  frustration         : frustrated | neutral | not_frustrated
  gave_too_much       : yes | partial | no   (did LLM hand over the answer?)
  asked_clarification : yes | no             (did student ask for clarification?)
  tutor_quality       : tutor | mixed | answer_generator

Requires: OPENAI_API_KEY env var.
Reads:    data/processed.csv
Writes:   data/labeled.csv (existing labels preserved; only new columns added)

Re-running is safe — rows that already have labels are skipped unless --force.
"""

import csv
import json
import os
import sys
import time
import argparse

try:
    from openai import OpenAI
except ImportError:
    print("openai package not found. Run: pip install openai")
    sys.exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "data", "processed.csv")
OUT_CSV = os.path.join(BASE, "data", "labeled.csv")

NEW_COLS = [
    "satisfaction",
    "confusion",
    "frustration",
    "gave_too_much",
    "asked_clarification",
    "tutor_quality",
]

SYSTEM_PROMPT = """You are an expert educational researcher analyzing student–LLM tutoring dialogues.
Given a student's message and the LLM's response, output ONLY a JSON object with these exact keys and allowed values:

{
  "satisfaction":        "satisfied" | "neutral" | "unsatisfied",
  "confusion":           "confused" | "unclear" | "not_confused",
  "frustration":         "frustrated" | "neutral" | "not_frustrated",
  "gave_too_much":       "yes" | "partial" | "no",
  "asked_clarification": "yes" | "no",
  "tutor_quality":       "tutor" | "mixed" | "answer_generator"
}

Definitions:
- satisfaction: does the student's phrasing suggest they are satisfied (got what they needed), unsatisfied (still stuck, asking again), or neutral?
- confusion: is the student's message itself expressing confusion or uncertainty?
- frustration: is there any sign of frustration or exasperation in the student's language?
- gave_too_much: did the LLM provide a direct/complete answer instead of guiding ("yes"), partly guide and partly answer ("partial"), or properly scaffold without giving away the answer ("no")?
- asked_clarification: does the student's message contain a request for clarification or further explanation?
- tutor_quality: does the LLM response behave like a good tutor (scaffolding, asking questions, guiding), an answer generator (just solves the problem), or mixed?

Output ONLY the JSON object, no prose.
"""


def build_user_message(student_text: str, llm_text: str) -> str:
    llm_excerpt = (llm_text[:600] + "…") if len(llm_text) > 600 else llm_text
    return (
        f"Student message:\n{student_text}\n\n"
        f"LLM response:\n{llm_excerpt if llm_excerpt else '(not available)'}"
    )


def call_gpt(client: OpenAI, student_text: str, llm_text: str, model: str = "gpt-4o-mini") -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(student_text, llm_text)},
        ],
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    return json.loads(text)


def validate_labels(labels: dict) -> dict:
    VALID = {
        "satisfaction":        {"satisfied", "neutral", "unsatisfied"},
        "confusion":           {"confused", "unclear", "not_confused"},
        "frustration":         {"frustrated", "neutral", "not_frustrated"},
        "gave_too_much":       {"yes", "partial", "no"},
        "asked_clarification": {"yes", "no"},
        "tutor_quality":       {"tutor", "mixed", "answer_generator"},
    }
    out = {}
    for col, valid_set in VALID.items():
        val = str(labels.get(col, "")).strip().lower()
        out[col] = val if val in valid_set else "unknown"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--force", action="store_true", help="Re-label all rows even if already labeled")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--limit", type=int, default=None, help="Only label first N rows")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: OPENAI_API_KEY not set. Export it or use --dry-run.")
        sys.exit(1)

    client = OpenAI(api_key=api_key) if not args.dry_run else None

    # Load existing output if any (for incremental runs)
    existing = {}
    if os.path.exists(OUT_CSV) and not args.force:
        with open(OUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["file"], row["conv_id"], row["turn_id"])
                existing[key] = row

    # Load input
    with open(IN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + [c for c in NEW_COLS if c not in rows[0]]
    out_rows = []
    labeled = 0
    skipped = 0

    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break

        key = (row["file"], row["conv_id"], row["turn_id"])

        # Reuse cached result if available and not forcing
        if key in existing and not args.force:
            cached = existing[key]
            if all(cached.get(c, "") not in ("", "unknown") for c in NEW_COLS):
                out_rows.append(cached)
                skipped += 1
                continue

        if args.dry_run:
            print(f"\n--- Row {i} ({row['file']} turn {row['turn_id']}) ---")
            print(build_user_message(row["student_text"], row["llm_text"])[:300])
            for c in NEW_COLS:
                row[c] = "dry_run"
            out_rows.append(row)
            continue

        # Call GPT
        try:
            raw_labels = call_gpt(client, row["student_text"], row["llm_text"], model=args.model)
            labels = validate_labels(raw_labels)
        except Exception as e:
            print(f"  Error on row {i} ({row['file']}): {e}", file=sys.stderr)
            labels = {c: "unknown" for c in NEW_COLS}

        row.update(labels)
        out_rows.append(row)
        labeled += 1
        print(f"[{i+1}/{len(rows)}] {row['file']} t{row['turn_id']}: {labels}")

        # Polite rate limiting
        time.sleep(0.3)

    # Write output
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. Labeled: {labeled}, skipped (cached): {skipped}. Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
