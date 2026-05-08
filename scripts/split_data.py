"""
Create a reproducible stratified 8:1:1 train/val/test split.

Rules enforced:
  - Classes with fewer than --min-class-size examples are dropped.
  - If a class has more than --max-per-class examples, a random subset
    of exactly --max-per-class is used (the rest are masked/excluded).
  - The three-way split uses sklearn StratifiedShuffleSplit, which
    guarantees at least 1 example per class in every split regardless
    of rounding.
  - All randomness is seeded; re-running produces the same split.

Output: data/splits.json  (indices into processed.csv row order)

Usage:
  python scripts/split_data.py               # default settings
  python scripts/split_data.py --max-per-class 24   # balance majority class
  python scripts/split_data.py --min-class-size 6   # include smaller classes
  python scripts/split_data.py --check              # just show what would happen
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV   = os.path.join(BASE, "data", "processed.csv")
OUT_JSON = os.path.join(BASE, "data", "splits.json")


def make_splits(
    df: pd.DataFrame,
    label_col: str,
    min_class_size: int = 6,
    max_per_class: int | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    # ── 1. Drop classes that are too small ─────────────────────────────────
    counts = df[label_col].value_counts()
    dropped = counts[counts < min_class_size]
    kept    = counts[counts >= min_class_size]

    if dropped.any():
        print(f"\nDropping {len(dropped)} class(es) with < {min_class_size} examples:")
        for cls, n in dropped.items():
            print(f"    {cls} (n={n})")

    df_work = df[df[label_col].isin(kept.index)].copy()
    df_work = df_work.reset_index(drop=True)

    # ── 2. Mask majority classes if max_per_class is set ───────────────────
    masked_out = []
    if max_per_class is not None:
        rows_keep = []
        for cls in kept.index:
            idx = df_work.index[df_work[label_col] == cls].tolist()
            if len(idx) > max_per_class:
                chosen = rng.choice(idx, size=max_per_class, replace=False).tolist()
                excluded = [i for i in idx if i not in chosen]
                masked_out.extend(excluded)
                rows_keep.extend(chosen)
                print(f"    {cls}: keeping {max_per_class}/{len(idx)} "
                      f"(masking {len(excluded)})")
            else:
                rows_keep.extend(idx)
        df_work = df_work.loc[sorted(rows_keep)].reset_index(drop=True)

    # ── 3. Stratified 8:1:1 split ──────────────────────────────────────────
    y = df_work[label_col].values

    # First cut: hold out test (10 %)
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    trainval_idx, test_idx = next(sss_test.split(df_work, y))

    # Second cut: hold out val from the remaining trainval (≈11.1 % of trainval → 10 % overall)
    val_frac_of_trainval = val_frac / (1 - test_frac)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_trainval, random_state=seed + 1
    )
    y_trainval = y[trainval_idx]
    rel_train_idx, rel_val_idx = next(sss_val.split(df_work.iloc[trainval_idx], y_trainval))
    train_idx = trainval_idx[rel_train_idx]
    val_idx   = trainval_idx[rel_val_idx]

    # ── 4. Report ──────────────────────────────────────────────────────────
    n_total = len(df_work)
    print(f"\n{'─'*55}")
    print(f"  Split summary  (seed={seed}, min_class={min_class_size}"
          + (f", max_per_class={max_per_class}" if max_per_class else "") + ")")
    print(f"  Total usable: {n_total}  "
          f"→  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"{'─'*55}")
    print(f"  {'Class':<30} {'Total':>5} {'Train':>6} {'Val':>5} {'Test':>5}")
    print(f"  {'─'*50}")
    class_splits = {}
    for cls in kept[kept.index.isin(df_work[label_col].unique())].index:
        tr = int((df_work.iloc[train_idx][label_col] == cls).sum())
        va = int((df_work.iloc[val_idx  ][label_col] == cls).sum())
        te = int((df_work.iloc[test_idx ][label_col] == cls).sum())
        n  = int((df_work[label_col] == cls).sum())
        flag = "  ⚠ val=0" if va == 0 else ("  ⚠ test=0" if te == 0 else "")
        print(f"  {cls:<30} {n:>5} {tr:>6} {va:>5} {te:>5}{flag}")
        class_splits[cls] = {"total": n, "train": tr, "val": va, "test": te}
    print(f"{'─'*55}")

    # ── 5. Build output ────────────────────────────────────────────────────
    # Translate df_work positions back to original processed.csv indices
    orig_indices = df_work.index.tolist()  # after reset_index this is 0..n-1
    # We need the ORIGINAL row numbers in df (before dropping/masking)
    # df_work was built from df after filtering; track original row numbers
    # (stored in a helper column added below — see caller)
    orig_train = df_work.iloc[train_idx]["_orig_idx"].tolist()
    orig_val   = df_work.iloc[val_idx  ]["_orig_idx"].tolist()
    orig_test  = df_work.iloc[test_idx ]["_orig_idx"].tolist()

    result = {
        "created":        str(date.today()),
        "seed":           seed,
        "min_class_size": min_class_size,
        "max_per_class":  max_per_class,
        "label_col":      label_col,
        "n_total":        n_total,
        "n_train":        len(train_idx),
        "n_val":          len(val_idx),
        "n_test":         len(test_idx),
        "classes":        class_splits,
        "train":          sorted(orig_train),
        "val":            sorted(orig_val),
        "test":           sorted(orig_test),
        "masked_out":     sorted(masked_out),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-class-size", type=int, default=6,
                        help="Drop classes with fewer than this many examples (default 6)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Mask examples from classes larger than this (default: no masking)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="goal", choices=["goal", "satisfaction"],
                        help="Which label column to split on")
    parser.add_argument("--check", action="store_true",
                        help="Print the split plan without writing splits.json")
    args = parser.parse_args()

    csv_path = IN_CSV
    if args.label == "satisfaction":
        csv_path = os.path.join(BASE, "data", "labeled.csv")
        if not os.path.exists(csv_path):
            sys.exit("labeled.csv not found — run llm_label.py first.")

    df = pd.read_csv(csv_path)
    df = df[df["student_text"].notna() & (df["student_text"].str.strip() != "")]
    df = df[df[args.label].notna() & ~df[args.label].isin(["", "Other", "unknown", "dry_run"])]
    df = df.reset_index(drop=True)
    df["_orig_idx"] = df.index  # preserve original row number

    splits = make_splits(
        df,
        label_col=args.label,
        min_class_size=args.min_class_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    if args.check:
        print("\n(--check mode: splits.json NOT written)")
        return

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
