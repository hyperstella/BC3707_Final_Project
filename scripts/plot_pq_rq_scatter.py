"""
Generate prompt-quality vs response-quality scatter plot with best-fit line.
Reads human-annotated rows from data/processed.csv.
Saves to Report/figures/prompt_response_scatter.png.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(BASE, "data", "processed.csv")
OUT_PNG = os.path.join(BASE, "Report", "figures", "prompt_response_scatter.png")

pq_vals, rq_vals = [], []
with open(IN_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("label_source", "").strip() != "human":
            continue
        pq = row.get("pq_score_norm", "").strip()
        rq = row.get("rq_score_norm", "").strip()
        if pq and rq and pq not in ("", "None") and rq not in ("", "None"):
            pq_vals.append(float(pq))
            rq_vals.append(float(rq))

pq = np.array(pq_vals)
rq = np.array(rq_vals)

slope, intercept, r, p_val, _ = stats.linregress(pq, rq)
x_line = np.linspace(pq.min(), pq.max(), 200)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(pq, rq, alpha=0.65, edgecolors="steelblue", facecolors="lightsteelblue",
           s=40, linewidths=0.6, zorder=3)
ax.plot(x_line, y_line, color="tomato", linewidth=1.5, label=f"OLS fit ($r={r:.3f}$, $p={p_val:.3f}$)")

ax.set_xlabel("Prompt quality (normalised)", fontsize=11)
ax.set_ylabel("Response quality (normalised)", fontsize=11)
ax.set_title("Prompt vs. Response Quality\n(64 human-annotated turns)", fontsize=11)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PNG}  (n={len(pq)}, r={r:.3f}, p={p_val:.4f})")
