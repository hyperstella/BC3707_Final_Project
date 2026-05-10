"""
Score LLM response quality for rows in processed.csv that are missing scores.

Adds per-turn scores for:
  rq_coherence   : 1–5  (logical organisation and clarity of the response)
  rq_correctness : 1–5  (factual / conceptual accuracy)
  rq_guidance    : 1–5  (pedagogical scaffolding vs. answer delivery)

Also recomputes:
  rq_score_raw   = 2*coherence + correctness + 3*guidance  (6–30)
  rq_score_norm  = (rq_score_raw - 6) / 24                 (0–1)

Rows that already have all three scores are skipped unless --force is passed.

Usage:
  python scripts/score_responses.py                 # score all missing rows
  python scripts/score_responses.py --dry-run       # print prompts, no API calls
  python scripts/score_responses.py --force         # re-score all rows
  python scripts/score_responses.py --limit 20      # only first 20 rows
  python scripts/score_responses.py --model gpt-4o  # use a different model

Requires: OPENAI_API_KEY env var.
"""

import argparse
import csv
import os
import sys
import time
import json

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found. Run: pip install openai")

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "data", "processed.csv")

SCORE_COLS = ["rq_coherence", "rq_correctness", "rq_guidance"]

SYSTEM_PROMPT = """You are an expert educational researcher evaluating LLM tutor responses.

Given a student's message and the LLM's response, score the response on three criteria, each on a scale of 1 to 5:

  - coherence (1–5): Is the response logically organised, easy to follow, and free of contradictions?
      1 = incoherent or disorganised
      3 = mostly clear with minor issues
      5 = exceptionally clear and well-structured

  - correctness (1–5): Is the response factually and conceptually accurate?
      1 = contains major factual errors
      3 = mostly correct with minor inaccuracies
      5 = fully accurate

  - guidance (1–5): Does the response scaffold the student's understanding rather than just delivering the answer?
      1 = gives the answer directly with no explanation or scaffolding
      3 = partial scaffolding — some explanation but still answer-heavy
      5 = excellent tutoring — guides the student through reasoning without giving away the answer

Respond with ONLY a JSON object with exactly these three keys:
  {"coherence": <int>, "correctness": <int>, "guidance": <int>}

Output ONLY the JSON object, no prose.
"""


def build_user_message(student_text: str, llm_text: str) -> str:
    llm_excerpt = (llm_text[:600] + "…") if len(llm_text) > 600 else llm_text
    return (
        f"Student message:\n{student_text.strip()}\n\n"
        f"LLM response:\n{llm_excerpt.strip() or '(not available)'}"
    )


def call_gpt(client: OpenAI, student_text: str, llm_text: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(student_text, llm_text)},
        ],
        temperature=0,
        max_tokens=50,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content.strip())


def validate(raw: dict) -> dict:
    out = {}
    for col in ["coherence", "correctness", "guidance"]:
        try:
            val = int(raw.get(col, 0))
            out[col] = max(1, min(5, val))
        except (TypeError, ValueError):
            out[col] = None
    return out


def compute_rq(scores: dict) -> tuple:
    c = scores.get("coherence")
    r = scores.get("correctness")
    g = scores.get("guidance")
    if all(v is not None for v in [c, r, g]):
        raw  = 2 * c + r + 3 * g
        norm = round((raw - 6) / 24, 4)
        return raw, norm
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Re-score rows that already have scores")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key) if not args.dry_run else None

    with open(IN_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    scored = skipped = 0

    for i, row in enumerate(rows):
        if args.limit and scored >= args.limit:
            break

        already_scored = all(
            row.get(f"rq_{c}", "").strip() not in ("", "None")
            for c in ["coherence", "correctness", "guidance"]
        )
        if already_scored and not args.force:
            skipped += 1
            continue

        student_text = row.get("student_text", "").strip()
        llm_text     = row.get("llm_text", "").strip()

        if not student_text or not llm_text:
            skipped += 1
            continue

        if args.dry_run:
            print(f"\n--- Row {i} ({row['file']} turn {row['turn_id']}) ---")
            print(build_user_message(student_text, llm_text)[:300])
            scored += 1
            continue

        try:
            raw_scores = call_gpt(client, student_text, llm_text, args.model)
            scores     = validate(raw_scores)
        except Exception as e:
            print(f"  [ERROR] row {i} ({row['file']}): {e}", file=sys.stderr)
            scored += 1
            continue

        rq_raw, rq_norm = compute_rq(scores)

        row["rq_coherence"]   = scores["coherence"]
        row["rq_correctness"] = scores["correctness"]
        row["rq_guidance"]    = scores["guidance"]
        row["rq_score_raw"]   = rq_raw
        row["rq_score_norm"]  = rq_norm

        print(f"[{i+1}/{len(rows)}] {row['file']} t{row['turn_id']}: "
              f"coherence={scores['coherence']} correctness={scores['correctness']} "
              f"guidance={scores['guidance']}  rq_norm={rq_norm}")

        scored += 1
        time.sleep(0.2)

    if args.dry_run:
        print(f"\n[dry-run] Would score {scored} rows.")
        return

    with open(IN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Scored: {scored}, skipped (already scored or no text): {skipped}.")
    print(f"Updated: {IN_CSV}")


if __name__ == "__main__":
    main()
