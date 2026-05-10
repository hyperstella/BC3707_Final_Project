"""
Generate repair outcome and repair behavior distribution figures.
Reads data/repair_labels.csv.
Saves to Report/figures/repair_outcomes.png and repair_behaviors.png.
"""

import csv
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "data", "repair_labels.csv")
OUT_DIR = os.path.join(BASE, "Report", "figures")

rows = list(csv.DictReader(open(IN_CSV, encoding="utf-8")))
n = len(rows)

# --- Repair Outcomes ---
outcome_order = [
    "Successful Repair",
    "Partial Repair",
    "Failed Repair",
    "Repeated Misconception",
]
outcome_counts = Counter(r["repair_outcome"] for r in rows)
oc = [outcome_counts.get(o, 0) for o in outcome_order]
oc_pct = [v / n * 100 for v in oc]
colors_out = ["#4caf50", "#ff9800", "#f44336", "#9c27b0"]

fig, ax = plt.subplots(figsize=(5.5, 3.4))
bars = ax.bar(outcome_order, oc, color=colors_out, edgecolor="white", linewidth=0.8)
for bar, cnt, pct in zip(bars, oc, oc_pct):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{cnt}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(f"Repair Outcome Distribution ($n={n}$ follow-up turns)", fontsize=10)
ax.set_ylim(0, max(oc) * 1.22)
ax.set_xticks(range(len(outcome_order)))
ax.set_xticklabels(outcome_order, fontsize=9, wrap=True)
ax.tick_params(axis="x", length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "repair_outcomes.png"), dpi=150, bbox_inches="tight")
print("Saved: repair_outcomes.png")
plt.close()

# --- Repair Behaviors ---
behavior_order = [
    "Changed strategy after follow-up",
    "Used step-by-step structure",
    "Used concrete example",
    "Used simplification",
    "Student showed further confusion afterward",
    "Used analogy or visual explanation",
]
all_behaviors = []
for r in rows:
    all_behaviors.extend([b.strip() for b in r["repair_behaviors"].split(";") if b.strip()])
bc = Counter(all_behaviors)
bvals = [bc.get(b, 0) for b in behavior_order]
bpct  = [v / n * 100 for v in bvals]

fig, ax = plt.subplots(figsize=(6.5, 3.8))
colors_beh = ["#1976d2"] * len(behavior_order)
y = np.arange(len(behavior_order))
bars = ax.barh(y, bvals, color=colors_beh, edgecolor="white", linewidth=0.6)
for bar, cnt, pct in zip(bars, bvals, bpct):
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{cnt} ({pct:.1f}%)", va="center", fontsize=8.5)

short_labels = [
    "Changed strategy",
    "Step-by-step structure",
    "Concrete example",
    "Simplification",
    "Further confusion afterward",
    "Analogy / visual",
]
ax.set_yticks(y)
ax.set_yticklabels(short_labels, fontsize=9)
ax.set_xlabel("Count", fontsize=10)
ax.set_title(f"Repair Behavior Frequency ($n={n}$ exchanges; multiple tags allowed)", fontsize=10)
ax.set_xlim(0, max(bvals) * 1.25)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "repair_behaviors.png"), dpi=150, bbox_inches="tight")
print("Saved: repair_behaviors.png")
plt.close()
