"""
Preprocess all dialogue JSON files into a flat CSV.

Handles:
  - Multiple JSON schema variants (per-turn vs conversation-level annotations)
  - Broken JSON (trailing commas, invalid escapes, control chars) via regex cleanup
  - Normalizes Goal/Scenario to canonical label sets
  - Outputs data/processed.csv with one row per labeled student turn
"""

import json
import re
import glob
import os
import csv
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "Final Dataset")
OUT_CSV = os.path.join(BASE, "data", "processed.csv")

# ── Canonical label maps ───────────────────────────────────────────────────────

GOAL_MAP = {
    # Answer Clarification
    "answer clarification": "Answer Clarification",
    "answer clarifcation": "Answer Clarification",
    "conceptual clarification": "Answer Clarification",
    "notation clarification": "Answer Clarification",
    "question clarification": "Answer Clarification",
    "correct or doubt llm's answer": "Answer Clarification",
    "simplification / shorter explanation": "Answer Clarification",

    # Concept Explanation
    "concept explanation": "Concept Explanation",
    "algorithm explanation": "Concept Explanation",
    "conceptual understanding": "Concept Explanation",
    "programming concepts": "Concept Explanation",
    "data structure explanation": "Concept Explanation",
    "backtracking explanation": "Concept Explanation",
    "dynamic programming explanation": "Concept Explanation",
    "dp base case explanation": "Concept Explanation",
    "proof explanation": "Concept Explanation",
    "simplification / intuition": "Concept Explanation",
    "example request": "Concept Explanation",
    "concrete example / demonstration": "Concept Explanation",

    # Step-by-Step Guidance
    "step-by-step guidance": "Step-by-Step Guidance",
    "strategy / rule": "Step-by-Step Guidance",
    "strategy / concept explanation": "Step-by-Step Guidance",
    "code modification guidance": "Step-by-Step Guidance",
    "project guidance": "Step-by-Step Guidance",
    "strategy or hint seeking": "Step-by-Step Guidance",

    # Direct Answer Generation
    "answer generation": "Direct Answer Generation",
    "direct answer generation": "Direct Answer Generation",

    # Debugging / Error Fixing
    "debugging": "Debugging / Error Fixing",
    "debugging / error fixing": "Debugging / Error Fixing",
    "debugging/error fixing": "Debugging / Error Fixing",

    # Step-by-Step Guidance variants
    "step by step guidance": "Step-by-Step Guidance",
    "step-by-step": "Step-by-Step Guidance",

    # Direct Answer Generation variants
    "answer generation (direct)": "Direct Answer Generation",
}

SCENARIO_MAP = {
    "assignment": "Assignment Help",
    "assignment help": "Assignment Help",
    "exam preparation": "Exam Preparation",
    "exam prep": "Exam Preparation",
    "lecture learning": "Lecture Learning",
    "lecture": "Lecture Learning",
    "project": "Project/Research Support",
    "project or research support": "Project/Research Support",
    "project/research support": "Project/Research Support",
    "research": "Project/Research Support",
    "not clear": "Not Clear",
}


def normalize_goal(raw: str) -> str:
    if not raw:
        return "Other"
    key = raw.strip().lower()
    # Long free-text scenarios from early annotation pass
    if len(key) > 60:
        return "Concept Explanation"
    return GOAL_MAP.get(key, "Other")


def normalize_scenario(raw: str) -> str:
    if not raw:
        return "Not Clear"
    key = raw.strip().lower()
    if len(key) > 60:
        return "Not Clear"
    return SCENARIO_MAP.get(key, "Not Clear")


# ── JSON repair helpers ────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Remove control characters (except \t \n \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def load_json_safe(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    for attempt in [raw, _repair_json(raw)]:
        for strict in [False, True]:
            try:
                return json.loads(attempt, strict=strict)
            except json.JSONDecodeError:
                pass

    # Last resort: try json5 if available
    try:
        import json5  # type: ignore
        return json5.loads(raw)
    except Exception:
        pass

    return None


# ── Sample extraction ──────────────────────────────────────────────────────────

def extract_samples(d: dict, filename: str) -> list[dict]:
    samples = []
    conv = d.get("conversation", [])
    raw_llm = d.get("llm", "")
    llm = raw_llm.strip() if raw_llm and raw_llm.strip() else "Gemini"
    llm = llm.upper() if llm.lower() in ("gpt", "chatgpt") else llm
    conv_id = str(d.get("conversation_id", ""))

    # Build per-turn lookup tables
    student_turns = {t["turn_id"]: t["content"] for t in conv if t.get("role") in ("student",)}
    llm_turns = {
        t["turn_id"]: t["content"]
        for t in conv
        if t.get("role") not in ("student",)
    }

    ann = d.get("annotations", {})

    def _extract_scores(ann_block: dict) -> dict:
        """
        Always recompute PQ and RQ from individual criteria so that files
        that stored a per-criterion value as the total (D3, D5, D7, D32, …)
        are corrected automatically.

        PQ  = Clarity + Specificity + Context   (max 15) → normalised /15
        RQ  = 2*Coherence + Correctness + 3*Guidance  (max 30) → normalised /30
        """
        pq_c = ann_block.get("prompt_quality", {})
        rq_c = ann_block.get("response_quality", {})
        pc = pq_c.get("criteria", {})
        rc = rq_c.get("criteria", {})

        clarity     = pc.get("Clarity of Intent")
        specificity = pc.get("Specificity & Precision") or pc.get("Specificity and Precision")
        context     = pc.get("Context Provision")
        coherence   = rc.get("Coherence")
        correctness = rc.get("Correctness")
        guidance    = rc.get("Guidance") or rc.get("Guidance Quality")

        if all(v is not None for v in [clarity, specificity, context]):
            pq_raw = clarity + specificity + context          # 3–15
            pq_norm = round((pq_raw - 3) / 12, 4)            # 0–1
        else:
            pq_raw = pq_norm = None

        if all(v is not None for v in [coherence, correctness, guidance]):
            rq_raw = 2 * coherence + correctness + 3 * guidance  # 6–30
            rq_norm = round((rq_raw - 6) / 24, 4)                # 0–1
        else:
            rq_raw = rq_norm = None

        return {
            "pq_clarity":     clarity,
            "pq_specificity": specificity,
            "pq_context":     context,
            "pq_score_raw":   pq_raw,
            "pq_score_norm":  pq_norm,
            "rq_coherence":   coherence,
            "rq_correctness": correctness,
            "rq_guidance":    guidance,
            "rq_score_raw":   rq_raw,
            "rq_score_norm":  rq_norm,
        }

    def _make_row(turn_id, text, llm_text, goal_raw, scenario_raw, scores: dict):
        goal = normalize_goal(goal_raw)
        scenario = normalize_scenario(scenario_raw)
        if goal == "Other" and not goal_raw:
            return None  # skip completely unannotated
        llm_clean = (llm_text or "").strip()
        # Summaries written by annotators start with "The response ..."
        is_summary = llm_clean.lower().startswith("the response")
        row = {
            "file": filename,
            "conv_id": conv_id,
            "turn_id": str(turn_id),
            "llm": llm,
            "student_text": text.strip(),
            "llm_text": llm_clean,
            "llm_text_is_summary": is_summary,
            "goal_raw": goal_raw,
            "goal": goal,
            "scenario_raw": scenario_raw,
            "scenario": scenario,
        }
        row.update(scores)
        return row

    if "turns" in ann:
        for t_ann in ann["turns"]:
            tid = t_ann.get("turn_id")
            intent = t_ann.get("intent", {})
            goal_raw = intent.get("Goal", "")
            scenario_raw = intent.get("Scenario", "")
            scores = _extract_scores(t_ann)
            text = student_turns.get(tid, "")
            llm_text = llm_turns.get(tid, "")
            if text:
                row = _make_row(tid, text, llm_text, goal_raw, scenario_raw, scores)
                if row:
                    samples.append(row)

    elif "intent" in ann:
        goal_raw = ann["intent"].get("Goal", "")
        scenario_raw = ann["intent"].get("Scenario", "")
        scores = _extract_scores(ann)
        # Use first student turn
        if student_turns:
            first_tid = min(student_turns.keys())
            text = student_turns[first_tid]
            llm_text = llm_turns.get(first_tid, "")
            row = _make_row(first_tid, text, llm_text, goal_raw, scenario_raw, scores)
            if row:
                samples.append(row)

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "file", "conv_id", "turn_id", "llm",
    "student_text", "llm_text", "llm_text_is_summary",
    "goal_raw", "goal", "scenario_raw", "scenario",
    # Individual criteria (1–5 each)
    "pq_clarity", "pq_specificity", "pq_context",
    "rq_coherence", "rq_correctness", "rq_guidance",
    # Aggregate scores: raw and 0–1 normalised
    "pq_score_raw", "pq_score_norm",
    "rq_score_raw", "rq_score_norm",
]


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    all_samples = []
    failed = []

    for path in files:
        d = load_json_safe(path)
        fname = os.path.basename(path)
        if d is None:
            failed.append(fname)
            continue
        samples = extract_samples(d, fname)
        all_samples.extend(samples)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_samples)

    print(f"Wrote {len(all_samples)} samples to {OUT_CSV}")

    # Label distribution
    from collections import Counter
    goals = Counter(s["goal"] for s in all_samples)
    scenarios = Counter(s["scenario"] for s in all_samples)
    print("\nGoal distribution:")
    for k, v in goals.most_common():
        print(f"  {k}: {v}")
    print("\nScenario distribution:")
    for k, v in scenarios.most_common():
        print(f"  {k}: {v}")

    if failed:
        print(f"\nSkipped {len(failed)} unparseable files: {', '.join(failed)}")


if __name__ == "__main__":
    main()
