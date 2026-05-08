"""
Mini-batch label verification for LLM-generated labels.

Randomly samples N rows from labeled.csv, shows you the student text,
LLM response, and each label, then lets you accept/reject/correct.

Results are written back to labeled.csv with a `verified` flag column.
Running multiple times safely accumulates verifications.

Usage:
  python scripts/verify_labels.py             # verify 10 random unverified rows
  python scripts/verify_labels.py --n 20      # verify 20 rows
  python scripts/verify_labels.py --all        # review all rows (including verified)
  python scripts/verify_labels.py --report     # print agreement stats, no interaction
"""

import argparse
import csv
import os
import random
import sys
import textwrap
from collections import defaultdict

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "data", "labeled.csv")

LLM_LABELS = [
    "satisfaction",
    "confusion",
    "frustration",
    "gave_too_much",
    "asked_clarification",
    "tutor_quality",
]

VALID_VALUES = {
    "satisfaction":        ["satisfied", "neutral", "unsatisfied"],
    "confusion":           ["confused", "unclear", "not_confused"],
    "frustration":         ["frustrated", "neutral", "not_frustrated"],
    "gave_too_much":       ["yes", "partial", "no"],
    "asked_clarification": ["yes", "no"],
    "tutor_quality":       ["tutor", "mixed", "answer_generator"],
}


# ── Display helpers ────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 90, indent: str = "  ") -> str:
    lines = text.replace("\\n", "\n").splitlines()
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width - len(indent)) or [""])
    return "\n".join(indent + l for l in wrapped)


def show_row(row: dict, idx: int, total: int):
    print(f"\n{'═'*70}")
    print(f"  [{idx}/{total}]  {row.get('file','')}  turn {row.get('turn_id','')}")
    print(f"{'─'*70}")
    print("  STUDENT:")
    print(_wrap(row.get("student_text", "(empty)")))
    llm = row.get("llm_text", "").strip()
    if llm:
        excerpt = llm[:500] + ("…" if len(llm) > 500 else "")
        print("  LLM RESPONSE:")
        print(_wrap(excerpt))
    print(f"{'─'*70}")
    print("  LLM LABELS:")
    for col in LLM_LABELS:
        val = row.get(col, "—")
        print(f"    {col:<22} {val}")
    print(f"{'─'*70}")


# ── Interaction ────────────────────────────────────────────────────────────────

def prompt_accept_reject(row: dict) -> tuple[str, dict]:
    """
    Returns (decision, corrections) where decision is 'accept'|'reject'|'correct'|'skip'.
    corrections is a dict of {col: new_value} for any labels changed.
    """
    while True:
        raw = input("  [a]ccept / [r]eject / [c]orrect label / [s]kip / [q]uit: ").strip().lower()
        if raw in ("q", "quit"):
            return "quit", {}
        if raw in ("s", "skip"):
            return "skip", {}
        if raw in ("a", "accept"):
            return "accept", {}
        if raw in ("r", "reject"):
            return "reject", {}
        if raw in ("c", "correct"):
            corrections = {}
            print("  Enter corrections (press Enter to keep current value):")
            for col in LLM_LABELS:
                current = row.get(col, "")
                opts = VALID_VALUES[col]
                shortlist = "  |  ".join(f"[{i+1}] {v}" for i, v in enumerate(opts))
                prompt = f"    {col} [{current}]  —  {shortlist}: "
                choice = input(prompt).strip()
                if not choice:
                    continue
                # Accept number or text
                if choice.isdigit() and 1 <= int(choice) <= len(opts):
                    corrections[col] = opts[int(choice) - 1]
                elif choice.lower() in opts:
                    corrections[col] = choice.lower()
                else:
                    print(f"      (unrecognised — keeping '{current}')")
            return "correct", corrections
        print("  Invalid input. Type a/r/c/s/q.")


# ── Agreement report ──────────────────────────────────────────────────────────

def print_report(rows: list[dict]):
    verified = [r for r in rows if r.get("verified") not in ("", None, "unverified")]
    if not verified:
        print("No verified rows found.")
        return

    total = len(verified)
    accepted = sum(1 for r in verified if r.get("verified") == "accept")
    rejected = sum(1 for r in verified if r.get("verified") == "reject")
    corrected = sum(1 for r in verified if r.get("verified") == "correct")

    print(f"\n{'═'*50}")
    print(f"  Verification report  ({total} rows reviewed)")
    print(f"{'─'*50}")
    print(f"  Accepted:  {accepted:>4}  ({100*accepted/total:.1f}%)")
    print(f"  Corrected: {corrected:>4}  ({100*corrected/total:.1f}%)")
    print(f"  Rejected:  {rejected:>4}  ({100*rejected/total:.1f}%)")
    print(f"{'─'*50}")

    # Per-label correction rate
    correction_counts = defaultdict(int)
    for r in verified:
        for col in LLM_LABELS:
            key = f"correction_{col}"
            if r.get(key, ""):
                correction_counts[col] += 1

    if correction_counts:
        print("  Per-label corrections:")
        for col in LLM_LABELS:
            n = correction_counts[col]
            pct = 100 * n / total
            bar = "█" * int(pct / 5)
            print(f"    {col:<22} {n:>3} ({pct:4.1f}%)  {bar}")

    print(f"{'═'*50}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of rows to verify")
    parser.add_argument("--all", action="store_true", help="Review all rows")
    parser.add_argument("--report", action="store_true", help="Print stats only, no interaction")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(IN_CSV):
        print(f"labeled.csv not found at {IN_CSV}")
        print("Run: python scripts/llm_label.py")
        sys.exit(1)

    with open(IN_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Add new columns if not present
    extra_cols = ["verified"] + [f"correction_{c}" for c in LLM_LABELS]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)
            for r in rows:
                r.setdefault(col, "")

    if args.report:
        print_report(rows)
        return

    # Select sample to verify
    if args.all:
        candidates = rows
    else:
        unverified = [r for r in rows if not r.get("verified")]
        if not unverified:
            print("All rows already verified. Use --all to re-review.")
            print_report(rows)
            return
        rng = random.Random(args.seed)
        candidates = rng.sample(unverified, min(args.n, len(unverified)))

    print(f"\nVerifying {len(candidates)} rows. Labels were generated by GPT.")
    print("Your decisions are written back to labeled.csv in real time.\n")

    # Build index for fast lookup
    row_index = {id(r): i for i, r in enumerate(rows)}

    verified_count = 0
    for i, row in enumerate(candidates, 1):
        show_row(row, i, len(candidates))
        decision, corrections = prompt_accept_reject(row)

        if decision == "quit":
            print("\nStopping early.")
            break
        if decision == "skip":
            continue

        row["verified"] = decision
        for col, new_val in corrections.items():
            row[f"correction_{col}"] = new_val

        verified_count += 1

        # Write back immediately so Ctrl-C doesn't lose work
        with open(IN_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nVerified {verified_count} rows.")
    print_report(rows)


if __name__ == "__main__":
    main()
