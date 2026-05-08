"""
Auto-label student-turn goals for conversations not yet in processed.csv.

For each JSON in 'Final Dataset/' that preprocess.py skipped (no annotations),
this script extracts every student turn, asks GPT to classify the goal, and
appends the resulting rows to data/processed.csv in the same schema.

Usage:
  python scripts/label_goals.py                  # label all missing files
  python scripts/label_goals.py --dry-run        # print prompts, no API calls
  python scripts/label_goals.py --limit 10       # only first 10 turns
  python scripts/label_goals.py --force          # re-label already-labeled files too
  python scripts/label_goals.py --model gpt-4o   # use a different model

Requires: OPENAI_API_KEY env var.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found. Run: pip install openai")

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE, "Final Dataset")
OUT_CSV     = os.path.join(BASE, "data", "processed.csv")

FIELDNAMES = [
    "file", "conv_id", "turn_id", "llm",
    "student_text", "llm_text", "llm_text_is_summary",
    "goal_raw", "goal", "scenario_raw", "scenario",
    "pq_clarity", "pq_specificity", "pq_context",
    "rq_coherence", "rq_correctness", "rq_guidance",
    "pq_score_raw", "pq_score_norm",
    "rq_score_raw", "rq_score_norm",
]

GOAL_CLASSES = [
    "Concept Explanation",
    "Answer Clarification",
    "Step-by-Step Guidance",
    "Direct Answer Generation",
    "Debugging / Error Fixing",
]

SYSTEM_PROMPT = """You are an expert educational researcher analyzing student–LLM tutoring dialogues.

Classify the student's message into exactly one of these goal categories:

  - Concept Explanation: student wants a concept, algorithm, theorem, or data structure explained
  - Answer Clarification: student wants to clarify, challenge, or correct a previous answer
  - Step-by-Step Guidance: student wants a walkthrough, strategy, or hint for solving a problem
  - Direct Answer Generation: student wants the answer or solution generated directly
  - Debugging / Error Fixing: student wants help finding or fixing a bug or code error

Respond with ONLY a JSON object with two keys:
  "goal": one of the five category names exactly as written above
  "scenario": one of "Assignment Help" | "Exam Preparation" | "Lecture Learning" | "Project/Research Support" | "Not Clear"

Output ONLY the JSON object, no prose.
"""


def build_few_shot_examples(n_per_class: int = 2) -> str:
    """Sample n examples per class from processed.csv and format as a prompt block."""
    if not os.path.exists(OUT_CSV):
        return ""
    with open(OUT_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("goal") in GOAL_CLASSES and r.get("student_text", "").strip()]

    by_class: dict[str, list] = {c: [] for c in GOAL_CLASSES}
    for r in rows:
        by_class[r["goal"]].append(r)

    lines = ["Here are labeled examples to guide your classification:\n"]
    for cls in GOAL_CLASSES:
        for ex in by_class[cls][:n_per_class]:
            text = ex["student_text"].strip()[:300]
            lines.append(f'Student: "{text}"\nGoal: {cls}\n')
    return "\n".join(lines)


def _repair(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def load_json(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    for attempt in [raw, _repair(raw)]:
        for strict in [False, True]:
            try:
                return json.loads(attempt, strict=strict)
            except json.JSONDecodeError:
                pass
    return None


def extract_turns(d: dict, filename: str) -> list[dict]:
    """Return list of {file, conv_id, turn_id, llm, student_text, llm_text} dicts."""
    conv = d.get("conversation", [])
    if not conv:
        return []

    raw_llm = d.get("llm") or d.get("model") or "Unknown"
    llm = raw_llm.strip()
    conv_id = str(d.get("conversation_id", os.path.splitext(filename)[0]))

    # Index turns by turn_id
    student_turns, llm_turns = {}, {}
    for t in conv:
        tid = t.get("turn_id")
        role = (t.get("role") or "").lower()
        content = t.get("content", "")
        if role == "student":
            student_turns[tid] = content
        else:
            llm_turns[tid] = content

    rows = []
    for tid, text in sorted(student_turns.items(), key=lambda x: x[0]):
        if not text or not str(text).strip():
            continue
        llm_text = llm_turns.get(tid, "")
        rows.append({
            "file": filename,
            "conv_id": conv_id,
            "turn_id": str(tid),
            "llm": llm,
            "student_text": str(text).strip(),
            "llm_text": str(llm_text).strip(),
        })
    return rows


def call_gpt(client: OpenAI, student_text: str, llm_text: str,
             model: str, few_shot_block: str = "") -> dict:
    llm_excerpt = (llm_text[:500] + "…") if len(llm_text) > 500 else llm_text
    user_msg = (
        f"{few_shot_block}\n"
        f"Now classify this new message:\n\n"
        f"Student message:\n{student_text}\n\n"
        f"LLM response:\n{llm_excerpt or '(not available)'}"
    ).strip()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=60,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content.strip())


def validate(raw: dict) -> tuple[str, str]:
    goal = str(raw.get("goal", "")).strip()
    scenario = str(raw.get("scenario", "")).strip()
    if goal not in GOAL_CLASSES:
        # fuzzy fallback: pick closest by substring
        lower = goal.lower()
        goal = next((c for c in GOAL_CLASSES if c.lower() in lower or lower in c.lower()),
                    "Other")
    valid_scenarios = {
        "Assignment Help", "Exam Preparation",
        "Lecture Learning", "Project/Research Support", "Not Clear",
    }
    if scenario not in valid_scenarios:
        scenario = "Not Clear"
    return goal, scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Re-label files already in processed.csv")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Stop after labeling this many turns")
    parser.add_argument("--shots",   type=int, default=2,
                        help="Few-shot examples per class from processed.csv (default: 2)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key) if not args.dry_run else None

    few_shot_block = build_few_shot_examples(n_per_class=args.shots)
    if few_shot_block:
        n_examples = few_shot_block.count("Goal:")
        print(f"Using {n_examples} few-shot examples ({args.shots} per class).")
    else:
        print("No existing labeled data found — using zero-shot prompting.")

    # Load existing rows so we know which files are already covered
    existing_rows = []
    existing_files: set[str] = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                existing_files.add(row["file"])

    all_json = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    print(f"Found {len(all_json)} JSON files. "
          f"{len(existing_files)} already in processed.csv.")

    new_rows = []
    total_labeled = 0
    total_skipped = 0

    for path in all_json:
        fname = os.path.basename(path)

        if fname in existing_files and not args.force:
            total_skipped += 1
            continue

        d = load_json(path)
        if d is None:
            print(f"  [SKIP] {fname} — failed to parse")
            continue

        # Skip files that have annotations (preprocess.py already handled them)
        if d.get("annotations") and not args.force:
            total_skipped += 1
            continue

        turns = extract_turns(d, fname)
        if not turns:
            print(f"  [SKIP] {fname} — no student turns")
            continue

        print(f"\n{fname}  ({len(turns)} student turns)")

        for turn in turns:
            if args.limit is not None and total_labeled >= args.limit:
                break

            if args.dry_run:
                print(f"  turn {turn['turn_id']}: {turn['student_text'][:80]}…")
                row = {**turn,
                       "llm_text_is_summary": False,
                       "goal_raw": "dry_run", "goal": "dry_run",
                       "scenario_raw": "", "scenario": "Not Clear",
                       **{c: "" for c in FIELDNAMES if c not in turn}}
                new_rows.append(row)
                total_labeled += 1
                continue

            try:
                raw = call_gpt(client, turn["student_text"], turn["llm_text"],
                               args.model, few_shot_block)
                goal, scenario = validate(raw)
            except Exception as e:
                print(f"  [ERROR] turn {turn['turn_id']}: {e}")
                goal, scenario = "Other", "Not Clear"

            llm_clean = turn["llm_text"]
            is_summary = llm_clean.lower().startswith("the response")

            row = {
                **turn,
                "llm_text_is_summary": is_summary,
                "goal_raw":     goal,
                "goal":         goal,
                "scenario_raw": scenario,
                "scenario":     scenario,
                # Score columns left empty — not available for unannotated files
                "pq_clarity": "", "pq_specificity": "", "pq_context": "",
                "rq_coherence": "", "rq_correctness": "", "rq_guidance": "",
                "pq_score_raw": "", "pq_score_norm": "",
                "rq_score_raw": "", "rq_score_norm": "",
            }
            new_rows.append(row)
            total_labeled += 1
            print(f"  turn {turn['turn_id']:>3}  goal={goal:<30}  scenario={scenario}")
            time.sleep(0.2)

        if args.limit is not None and total_labeled >= args.limit:
            print(f"\nReached --limit {args.limit}, stopping.")
            break

    if args.dry_run:
        print(f"\n[dry-run] Would append {len(new_rows)} rows to {OUT_CSV}")
        return

    if not new_rows:
        print("\nNothing new to add.")
        return

    # Append to processed.csv
    all_rows = existing_rows + new_rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Added {len(new_rows)} new rows → {OUT_CSV}  "
          f"(total: {len(all_rows)})")
    print(f"Skipped {total_skipped} already-processed files.")


if __name__ == "__main__":
    main()
