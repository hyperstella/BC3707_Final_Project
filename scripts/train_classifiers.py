"""
Classifier training for the Student–LLM Tutoring dataset.

Trains and evaluates three models:
  1. TF-IDF + Logistic Regression (interpretable baseline)
  2. Sentence-Transformer + MLP  (main neural model)
  3. TextCNN with learned embeddings (report-specified CNN)

Each model is evaluated with stratified k-fold cross-validation.
Metrics: accuracy, macro-F1, confusion matrix.

Target tasks:
  A) Goal prediction       (5 classes from data/processed.csv)
  B) Satisfaction prediction (3 classes from data/labeled.csv, if available)

Usage:
  python scripts/train_classifiers.py            # runs both tasks
  python scripts/train_classifiers.py --task goal
  python scripts/train_classifiers.py --task satisfaction
  python scripts/train_classifiers.py --no-cnn   # skip TextCNN (faster)
"""

import argparse
import json
import os
import sys
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS     = os.path.join(BASE, "results")
SPLITS_JSON = os.path.join(BASE, "data", "splits.json")
os.makedirs(RESULTS, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────────

SCORE_COLS = [
    "pq_clarity", "pq_specificity", "pq_context",
    "rq_coherence", "rq_correctness", "rq_guidance",
    "pq_score_norm", "rq_score_norm",
]


def _filter_df(df: pd.DataFrame, label_col: str, min_class_size: int) -> pd.DataFrame:
    df = df[df["student_text"].notna() & (df["student_text"].str.strip() != "")]
    df = df[df[label_col].notna() & ~df[label_col].isin(["", "Other", "unknown", "dry_run"])]
    counts = df[label_col].value_counts()
    dropped = counts[counts < min_class_size]
    if not dropped.empty:
        print(f"  Dropping classes with <{min_class_size} examples: "
              + ", ".join(f"{k}({v})" for k, v in dropped.items()))
    keep = counts[counts >= min_class_size].index
    return df[df[label_col].isin(keep)].copy()


def majority_baseline(labels) -> float:
    c = Counter(labels)
    return c.most_common(1)[0][1] / len(labels)


def load_splits() -> dict | None:
    if os.path.exists(SPLITS_JSON):
        with open(SPLITS_JSON) as f:
            return json.load(f)
    return None


def _load_raw_for_splits(csv_path: str, label_col: str) -> pd.DataFrame:
    """Load and filter identically to split_data.py so indices match splits.json."""
    df = pd.read_csv(csv_path)
    df = df[df["student_text"].notna() & (df["student_text"].str.strip() != "")]
    df = df[df[label_col].notna() & ~df[label_col].isin(["", "Other", "unknown", "dry_run"])]
    return df.reset_index(drop=True)


def get_split_dfs(df: pd.DataFrame, splits_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df.index.isin(set(splits_data["train"]))].copy().reset_index(drop=True)
    val   = df[df.index.isin(set(splits_data["val"]))].copy().reset_index(drop=True)
    test  = df[df.index.isin(set(splits_data["test"]))].copy().reset_index(drop=True)
    return train, val, test


def load_goal_data(min_class_size: int = 5) -> pd.DataFrame:
    """min_class_size=5 ensures every class has at least one example per LOOCV fold."""
    path = os.path.join(BASE, "data", "processed.csv")
    if not os.path.exists(path):
        sys.exit("processed.csv not found — run preprocess.py first.")
    df = _filter_df(pd.read_csv(path), "goal", min_class_size)
    maj = majority_baseline(df["goal"].tolist())
    print(f"\nGoal task: {len(df)} samples, {df['goal'].nunique()} classes  "
          f"(majority baseline: {maj:.3f})")
    print(df["goal"].value_counts().to_string())
    return df


def load_satisfaction_data(min_class_size: int = 5) -> pd.DataFrame:
    path = os.path.join(BASE, "data", "labeled.csv")
    if not os.path.exists(path):
        sys.exit("labeled.csv not found — run llm_label.py first.")
    df = _filter_df(pd.read_csv(path), "satisfaction", min_class_size)
    maj = majority_baseline(df["satisfaction"].tolist())
    print(f"\nSatisfaction task: {len(df)} samples, {df['satisfaction'].nunique()} classes  "
          f"(majority baseline: {maj:.3f})")
    print(df["satisfaction"].value_counts().to_string())
    return df


def get_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Return imputed, scaled numeric score matrix (n_samples × len(SCORE_COLS))."""
    present = [c for c in SCORE_COLS if c in df.columns]
    X_num = df[present].values.astype(float)
    X_num = SimpleImputer(strategy="mean").fit_transform(X_num)
    X_num = StandardScaler().fit_transform(X_num)
    return X_num


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def print_results(name, accs, f1s, majority_acc: float = None):
    print(f"\n{'─'*50}")
    print(f"  {name}")
    acc_mean = np.mean(accs)
    print(f"  Accuracy:  {acc_mean:.3f} ± {np.std(accs):.3f}", end="")
    if majority_acc is not None:
        delta = acc_mean - majority_acc
        print(f"  (majority baseline: {majority_acc:.3f}, Δ={delta:+.3f})", end="")
    print()
    print(f"  Macro-F1:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


# ── sklearn helper: wraps a column of a DataFrame as a text list ───────────────

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col="student_text"):
        self.col = col
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.col].fillna("").tolist()


class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols or SCORE_COLS
    def fit(self, X, y=None): return self
    def transform(self, X):
        present = [c for c in self.cols if c in X.columns]
        return X[present].values.astype(float)


# ── Model 1: TF-IDF + Logistic Regression ─────────────────────────────────────

def run_tfidf_logreg(df: pd.DataFrame, label_col: str, task_name: str, n_splits: int = 5):
    print(f"\n[TF-IDF + LogReg] task={task_name}  (LOOCV, n={len(df)})")
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    class_names = le.classes_
    maj = majority_baseline(df[label_col].tolist())

    # LOOCV: each sample is the test set once — maximises train data for tiny datasets
    loo = LeaveOneOut()
    all_true, all_pred = [], []

    from scipy.sparse import hstack
    import scipy.sparse as sp

    num_cols_present = [c for c in SCORE_COLS if c in df.columns]
    has_scores = len(num_cols_present) > 0

    text_pipe = Pipeline([
        ("sel", TextSelector()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ])
    num_pipe = Pipeline([
        ("sel", NumericSelector(num_cols_present)),
        ("imp", SimpleImputer(strategy="mean")),
        ("scl", StandardScaler(with_mean=False)),
    ]) if has_scores else None

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")

    for train_idx, val_idx in loo.split(df):
        df_tr = df.iloc[train_idx]
        df_v  = df.iloc[val_idx]

        X_tr = text_pipe.fit_transform(df_tr)
        X_v  = text_pipe.transform(df_v)

        if has_scores:
            Xn_tr = sp.csr_matrix(num_pipe.fit_transform(df_tr))
            Xn_v  = sp.csr_matrix(num_pipe.transform(df_v))
            X_train = hstack([X_tr, Xn_tr])
            X_val   = hstack([X_v,  Xn_v])
        else:
            X_train, X_val = X_tr, X_v

        clf.fit(X_train, y[train_idx])
        all_true.append(y[val_idx[0]])
        all_pred.append(clf.predict(X_val)[0])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc  = accuracy_score(all_true, all_pred)
    mf1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(all_true, all_pred, labels=range(len(class_names)))

    label = "TF-IDF + LogReg (text+scores)" if has_scores else "TF-IDF + LogReg"
    # LOOCV gives a single point estimate, no std
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  Accuracy:  {acc:.3f}  (majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(f"\n  Per-class report:")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    plot_confusion_matrix(
        cm, class_names,
        f"TF-IDF + LogReg — {task_name} (LOOCV)",
        os.path.join(RESULTS, f"{task_name}_tfidf_cm.png"),
    )
    return {"accuracy": acc, "f1": mf1}


# ── Model 2: Sentence-Transformer + MLP ───────────────────────────────────────

def get_sentence_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print("  Loading sentence-transformer model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, n_classes, epochs=50, lr=1e-3):
    device = torch.device("cpu")
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

    model = MLP(X_train.shape[1], 256, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [len(y_train) / (n_classes * np.bincount(y_train, minlength=n_classes)[c] + 1e-6)
             for c in range(n_classes)],
            dtype=torch.float32
        )
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_v)
        preds = logits.argmax(dim=1).numpy()

    return preds


def _train_mlp_es(X_train, y_train, X_val, y_val, X_test, n_classes,
                  epochs=120, lr=1e-3, patience=20):
    """MLP with early stopping on val; returns predictions on X_test."""
    device = torch.device("cpu")
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xv  = torch.tensor(X_val,   dtype=torch.float32)
    yv  = torch.tensor(y_val,   dtype=torch.long)
    Xte = torch.tensor(X_test,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=8, shuffle=True)
    model = MLP(X_train.shape[1], 256, n_classes).to(device)
    w = torch.tensor(
        [len(y_train) / (n_classes * max(1, np.bincount(y_train, minlength=n_classes)[c]))
         for c in range(n_classes)], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_state, no_imp = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()
        if vl < best_val:
            best_val, no_imp = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xte).argmax(dim=1).numpy()


def run_sentence_mlp(df: pd.DataFrame, label_col: str, task_name: str, n_splits: int = 5):
    print(f"\n[Sentence-Transformer + MLP] task={task_name}  (LOOCV, n={len(df)})")
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    class_names = le.classes_
    maj = majority_baseline(df[label_col].tolist())

    texts = df["student_text"].tolist()
    text_emb = get_sentence_embeddings(texts)

    num_cols_present = [c for c in SCORE_COLS if c in df.columns]
    X_num_raw = df[num_cols_present].values.astype(float) if num_cols_present else None

    loo = LeaveOneOut()
    all_true, all_pred, cms = [], [], []

    for train_idx, val_idx in loo.split(text_emb):
        X_text_tr = text_emb[train_idx]
        X_text_v  = text_emb[val_idx]

        if X_num_raw is not None:
            imp = SimpleImputer(strategy="mean")
            scl = StandardScaler()
            Xn_tr = scl.fit_transform(imp.fit_transform(X_num_raw[train_idx]))
            Xn_v  = scl.transform(imp.transform(X_num_raw[val_idx]))
            X_train = np.hstack([X_text_tr, Xn_tr])
            X_val   = np.hstack([X_text_v,  Xn_v])
        else:
            X_train, X_val = X_text_tr, X_text_v

        y_train = y[train_idx]
        preds = train_mlp(X_train, y_train, X_val, y[val_idx], n_classes=len(class_names))
        all_true.append(y[val_idx[0]])
        all_pred.append(preds[0])
        cms.append(confusion_matrix([y[val_idx[0]]], [preds[0]], labels=range(len(class_names))))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = accuracy_score(all_true, all_pred)
    mf1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    label = "Sentence-MLP (text+scores)" if X_num_raw is not None else "Sentence-MLP"
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  Accuracy:  {acc:.3f}  (majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(f"\n  Per-class report:")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    agg_cm = np.sum(cms, axis=0)
    plot_confusion_matrix(
        agg_cm, class_names,
        f"Sentence-MLP — {task_name} (LOOCV)",
        os.path.join(RESULTS, f"{task_name}_sentence_mlp_cm.png"),
    )
    return {"accuracy": acc, "f1": mf1}


# ── Model 3: TextCNN ───────────────────────────────────────────────────────────

def build_vocab(texts: list[str], max_vocab: int = 5000) -> dict:
    from collections import Counter
    tokens_all = [tok for t in texts for tok in t.lower().split()]
    most_common = Counter(tokens_all).most_common(max_vocab - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def texts_to_ids(texts: list[str], vocab: dict, max_len: int = 64) -> np.ndarray:
    result = []
    unk = vocab["<UNK>"]
    pad = vocab["<PAD>"]
    for text in texts:
        ids = [vocab.get(w, unk) for w in text.lower().split()][:max_len]
        ids += [pad] * (max_len - len(ids))
        result.append(ids)
    return np.array(result, dtype=np.int64)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_sizes=(3, 4, 5),
                 n_filters=128, dropout=0.5, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))              # (batch, n_filters, L)
            c = c.max(dim=2).values                # (batch, n_filters)
            pooled.append(c)
        cat = torch.cat(pooled, dim=1)             # (batch, n_filters * len(filter_sizes))
        return self.fc(self.dropout(cat))


def train_cnn(X_train, y_train, X_val, y_val, vocab_size, n_classes, epochs=60, lr=5e-4):
    device = torch.device("cpu")
    X_tr = torch.tensor(X_train, dtype=torch.long)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.long)
    y_v  = torch.tensor(y_val,   dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

    model = TextCNN(vocab_size=vocab_size, embed_dim=64, n_classes=n_classes).to(device)
    weights = torch.tensor(
        [len(y_train) / (n_classes * np.bincount(y_train, minlength=n_classes)[c] + 1e-6)
         for c in range(n_classes)],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_v).argmax(dim=1).numpy()

    return preds


def _train_cnn_es(X_train, y_train, X_val, y_val, X_test, vocab_size, n_classes,
                  epochs=120, lr=5e-4, patience=20):
    """TextCNN with early stopping on val; returns predictions on X_test."""
    device = torch.device("cpu")
    Xtr = torch.tensor(X_train, dtype=torch.long)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xv  = torch.tensor(X_val,   dtype=torch.long)
    yv  = torch.tensor(y_val,   dtype=torch.long)
    Xte = torch.tensor(X_test,  dtype=torch.long)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=8, shuffle=True)
    model = TextCNN(vocab_size=vocab_size, embed_dim=64, n_classes=n_classes).to(device)
    w = torch.tensor(
        [len(y_train) / (n_classes * max(1, np.bincount(y_train, minlength=n_classes)[c]))
         for c in range(n_classes)], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_state, no_imp = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()
        if vl < best_val:
            best_val, no_imp = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xte).argmax(dim=1).numpy()


def run_textcnn(df: pd.DataFrame, label_col: str, task_name: str, n_splits: int = 5):
    print(f"\n[TextCNN — text only] task={task_name}  (LOOCV, n={len(df)})")
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    class_names = le.classes_
    maj = majority_baseline(df[label_col].tolist())

    texts = df["student_text"].tolist()
    vocab = build_vocab(texts)
    X = texts_to_ids(texts, vocab)

    loo = LeaveOneOut()
    all_true, all_pred, cms = [], [], []

    for train_idx, val_idx in loo.split(X):
        preds = train_cnn(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            vocab_size=len(vocab), n_classes=len(class_names),
        )
        all_true.append(y[val_idx[0]])
        all_pred.append(preds[0])
        cms.append(confusion_matrix([y[val_idx[0]]], [preds[0]], labels=range(len(class_names))))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = accuracy_score(all_true, all_pred)
    mf1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  TextCNN (text only)")
    print(f"  Accuracy:  {acc:.3f}  (majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(f"\n  Per-class report:")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    agg_cm = np.sum(cms, axis=0)
    plot_confusion_matrix(
        agg_cm, class_names,
        f"TextCNN — {task_name} (LOOCV)",
        os.path.join(RESULTS, f"{task_name}_textcnn_cm.png"),
    )
    return {"accuracy": acc, "f1": mf1}


# ── Fixed-split (8:1:1) evaluation ────────────────────────────────────────────

def _majority_acc_fixed(df_train, df_test, label_col):
    train_majority = Counter(df_train[label_col]).most_common(1)[0][0]
    return (df_test[label_col] == train_majority).mean()


def run_tfidf_fixed(df_train, df_val, df_test, label_col, task_name):
    from scipy.sparse import hstack
    import scipy.sparse as sp

    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    le = LabelEncoder().fit(all_df[label_col])
    class_names = le.classes_
    maj = _majority_acc_fixed(df_train, df_test, label_col)

    num_cols_present = [c for c in SCORE_COLS if c in df_train.columns]
    has_scores = bool(num_cols_present)

    text_pipe = Pipeline([
        ("sel", TextSelector()),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ])
    num_pipe = Pipeline([
        ("sel", NumericSelector(num_cols_present)),
        ("imp", SimpleImputer(strategy="mean")),
        ("scl", StandardScaler(with_mean=False)),
    ]) if has_scores else None
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")

    X_tr = text_pipe.fit_transform(df_train)
    X_te = text_pipe.transform(df_test)
    y_tr = le.transform(df_train[label_col])
    y_te = le.transform(df_test[label_col])

    if has_scores:
        X_tr = hstack([X_tr, sp.csr_matrix(num_pipe.fit_transform(df_train))])
        X_te = hstack([X_te, sp.csr_matrix(num_pipe.transform(df_test))])

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(y_te, preds, labels=range(len(class_names)))

    lbl = "TF-IDF + LogReg (text+scores)" if has_scores else "TF-IDF + LogReg"
    print(f"\n{'─'*55}")
    print(f"  {lbl}  [fixed split, test n={len(df_test)}]")
    print(f"  Accuracy:  {acc:.3f}  (train-majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_te, preds, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"TF-IDF + LogReg — {task_name} (test)",
                          os.path.join(RESULTS, f"{task_name}_tfidf_cm.png"))
    return {"accuracy": acc, "f1": mf1}


def run_sentence_mlp_fixed(df_train, df_val, df_test, label_col, task_name):
    print(f"\n[Sentence-MLP fixed split] task={task_name}")
    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    le = LabelEncoder().fit(all_df[label_col])
    class_names = le.classes_
    maj = _majority_acc_fixed(df_train, df_test, label_col)

    all_texts = (df_train["student_text"].tolist() + df_val["student_text"].tolist()
                 + df_test["student_text"].tolist())
    all_emb = get_sentence_embeddings(all_texts)
    n_tr, n_v = len(df_train), len(df_val)
    emb_tr = all_emb[:n_tr]
    emb_v  = all_emb[n_tr:n_tr + n_v]
    emb_te = all_emb[n_tr + n_v:]

    num_cols = [c for c in SCORE_COLS if c in df_train.columns]
    if num_cols:
        imp = SimpleImputer(strategy="mean")
        scl = StandardScaler()
        Xn_tr = scl.fit_transform(imp.fit_transform(df_train[num_cols].values.astype(float)))
        Xn_v  = scl.transform(imp.transform(df_val[num_cols].values.astype(float)))
        Xn_te = scl.transform(imp.transform(df_test[num_cols].values.astype(float)))
        X_tr = np.hstack([emb_tr, Xn_tr])
        X_v  = np.hstack([emb_v,  Xn_v])
        X_te = np.hstack([emb_te, Xn_te])
    else:
        X_tr, X_v, X_te = emb_tr, emb_v, emb_te

    y_tr = le.transform(df_train[label_col])
    y_v  = le.transform(df_val[label_col])
    y_te = le.transform(df_test[label_col])
    preds = _train_mlp_es(X_tr, y_tr, X_v, y_v, X_te, n_classes=len(class_names))

    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(y_te, preds, labels=range(len(class_names)))

    lbl = "Sentence-MLP (text+scores)" if num_cols else "Sentence-MLP"
    print(f"\n{'─'*55}")
    print(f"  {lbl}  [fixed split, test n={len(df_test)}]")
    print(f"  Accuracy:  {acc:.3f}  (train-majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_te, preds, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"Sentence-MLP — {task_name} (test)",
                          os.path.join(RESULTS, f"{task_name}_sentence_mlp_cm.png"))
    return {"accuracy": acc, "f1": mf1}


def run_textcnn_fixed(df_train, df_val, df_test, label_col, task_name):
    print(f"\n[TextCNN fixed split] task={task_name}")
    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    le = LabelEncoder().fit(all_df[label_col])
    class_names = le.classes_
    maj = _majority_acc_fixed(df_train, df_test, label_col)

    all_texts = (df_train["student_text"].tolist() + df_val["student_text"].tolist()
                 + df_test["student_text"].tolist())
    vocab = build_vocab(all_texts)
    n_tr, n_v = len(df_train), len(df_val)
    X_all = texts_to_ids(all_texts, vocab)
    X_tr, X_v, X_te = X_all[:n_tr], X_all[n_tr:n_tr+n_v], X_all[n_tr+n_v:]

    y_tr = le.transform(df_train[label_col])
    y_v  = le.transform(df_val[label_col])
    y_te = le.transform(df_test[label_col])
    preds = _train_cnn_es(X_tr, y_tr, X_v, y_v, X_te, len(vocab), n_classes=len(class_names))

    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(y_te, preds, labels=range(len(class_names)))

    print(f"\n{'─'*55}")
    print(f"  TextCNN (text only)  [fixed split, test n={len(df_test)}]")
    print(f"  Accuracy:  {acc:.3f}  (train-majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_te, preds, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"TextCNN — {task_name} (test)",
                          os.path.join(RESULTS, f"{task_name}_textcnn_cm.png"))
    return {"accuracy": acc, "f1": mf1}


# ── GPT-4 few-shot classification ─────────────────────────────────────────────

GOAL_DESCRIPTIONS = {
    "Concept Explanation":      "asks for an explanation of a concept, algorithm, data structure, or theory",
    "Answer Clarification":     "asks for clarification or correction of a previous answer or explanation",
    "Step-by-Step Guidance":    "requests step-by-step guidance, strategy, or a problem-solving walkthrough",
    "Direct Answer Generation": "requests a direct solution or answer to be generated",
    "Debugging / Error Fixing": "asks for help finding or fixing a bug or code error",
}
SAT_DESCRIPTIONS = {
    "satisfied":   "seems satisfied and understood the response",
    "neutral":     "has a neutral reaction to the response",
    "unsatisfied": "seems unsatisfied, confused, or frustrated with the response",
}


def _gpt4_classify_batch(texts, few_shot_rows, label_col, class_names, model="gpt-4o-mini"):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    descs = GOAL_DESCRIPTIONS if label_col == "goal" else SAT_DESCRIPTIONS
    class_lines = "\n".join(f"  - {c}: student {descs.get(c, c)}" for c in class_names)
    example_block = ""
    for row in few_shot_rows:
        example_block += f"Student: {row['student_text'].strip()[:400]}\nCategory: {row[label_col]}\n\n"
    system = (
        "You are a classifier. Classify student messages sent to an AI tutor.\n"
        f"Categories:\n{class_lines}\n"
        "Respond with exactly one category name, nothing else."
    )

    preds = []
    for text in texts:
        user_msg = example_block + f"Student: {text.strip()[:400]}\nCategory:"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                           {"role": "user",   "content": user_msg}],
                max_tokens=20, temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            pred = next((c for c in class_names if c.lower() in raw.lower() or raw.lower() in c.lower()), class_names[0])
        except Exception as e:
            print(f"    GPT-4 error: {e}")
            pred = class_names[0]
        preds.append(pred)
    return preds


def run_gpt4_fixed(df_train, df_val, df_test, label_col, task_name, n_shots=3, model="gpt-4o-mini"):
    print(f"\n[GPT-4 few-shot] task={task_name}  model={model}  {n_shots} shots/class")
    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    le = LabelEncoder().fit(all_df[label_col])
    class_names = list(le.classes_)
    maj = _majority_acc_fixed(df_train, df_test, label_col)

    few_shot = []
    for cls in class_names:
        few_shot.extend(df_train[df_train[label_col] == cls].head(n_shots).to_dict("records"))

    preds_raw = _gpt4_classify_batch(df_test["student_text"].tolist(), few_shot, label_col, class_names, model)
    preds = np.array([le.transform([p])[0] if p in class_names else 0 for p in preds_raw])
    y_te  = le.transform(df_test[label_col])

    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(y_te, preds, labels=range(len(class_names)))
    print(f"\n{'─'*55}")
    print(f"  GPT-4 few-shot ({model})  [fixed split, test n={len(df_test)}]")
    print(f"  Accuracy:  {acc:.3f}  (train-majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(classification_report(y_te, preds, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"GPT-4 few-shot — {task_name} (test)",
                          os.path.join(RESULTS, f"{task_name}_gpt4_cm.png"))
    return {"accuracy": acc, "f1": mf1}


def run_gpt4_loocv(df, label_col, task_name, n_shots=3, model="gpt-4o-mini"):
    print(f"\n[GPT-4 few-shot LOOCV] task={task_name}  model={model}  {n_shots} shots/class")
    le = LabelEncoder().fit(df[label_col])
    class_names = list(le.classes_)
    maj = majority_baseline(df[label_col].tolist())

    all_true, all_pred = [], []
    texts  = df["student_text"].tolist()
    labels = df[label_col].tolist()

    for i, (train_idx, val_idx) in enumerate(LeaveOneOut().split(df)):
        df_tr = df.iloc[train_idx]
        few_shot = []
        for cls in class_names:
            few_shot.extend(df_tr[df_tr[label_col] == cls].head(n_shots).to_dict("records"))
        pred_raw = _gpt4_classify_batch([texts[val_idx[0]]], few_shot, label_col, class_names, model)[0]
        pred = pred_raw if pred_raw in class_names else class_names[0]
        all_true.append(le.transform([labels[val_idx[0]]])[0])
        all_pred.append(le.transform([pred])[0])
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(df)} done…")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = accuracy_score(all_true, all_pred)
    mf1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(all_true, all_pred, labels=range(len(class_names)))
    print(f"\n{'─'*55}")
    print(f"  GPT-4 few-shot ({model})  [LOOCV, n={len(df)}]")
    print(f"  Accuracy:  {acc:.3f}  (majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"GPT-4 few-shot — {task_name} (LOOCV)",
                          os.path.join(RESULTS, f"{task_name}_gpt4_cm.png"))
    return {"accuracy": acc, "f1": mf1}


# ── Zero-shot NLI classification ───────────────────────────────────────────────

GOAL_HYPOTHESES = {
    "Concept Explanation":      "The student wants a concept, algorithm, or theory explained",
    "Answer Clarification":     "The student wants clarification or correction of a previous answer",
    "Step-by-Step Guidance":    "The student wants step-by-step guidance on how to solve a problem",
    "Direct Answer Generation": "The student wants a direct solution or answer generated",
    "Debugging / Error Fixing": "The student wants help debugging code or fixing an error",
}
SAT_HYPOTHESES = {
    "satisfied":   "The student is satisfied with the response they received",
    "neutral":     "The student has a neutral reaction to the response",
    "unsatisfied": "The student is unsatisfied or confused with the response",
}


def _nli_predict(texts, hypotheses_map, class_names):
    from transformers import pipeline as hf_pipeline
    print("  Loading NLI model (facebook/bart-large-mnli) — ~1.6 GB on first run…")
    nli = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    hypotheses = [hypotheses_map.get(c, c) for c in class_names]
    preds = []
    for text in texts:
        result = nli(text[:512], candidate_labels=hypotheses, multi_label=False)
        best = result["labels"][0]
        preds.append(class_names[hypotheses.index(best)])
    return preds


def run_zeroshot_nli_fixed(df_train, df_val, df_test, label_col, task_name):
    print(f"\n[Zero-shot NLI] task={task_name}")
    all_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    le = LabelEncoder().fit(all_df[label_col])
    class_names = list(le.classes_)
    maj = _majority_acc_fixed(df_train, df_test, label_col)
    hyp_map = GOAL_HYPOTHESES if label_col == "goal" else SAT_HYPOTHESES

    preds = le.transform(_nli_predict(df_test["student_text"].tolist(), hyp_map, class_names))
    y_te  = le.transform(df_test[label_col])
    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(y_te, preds, labels=range(len(class_names)))
    print(f"\n{'─'*55}")
    print(f"  Zero-shot NLI (bart-large-mnli)  [fixed split, test n={len(df_test)}]")
    print(f"  Accuracy:  {acc:.3f}  (train-majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(classification_report(y_te, preds, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"Zero-shot NLI — {task_name} (test)",
                          os.path.join(RESULTS, f"{task_name}_nli_cm.png"))
    return {"accuracy": acc, "f1": mf1}


def run_zeroshot_nli_loocv(df, label_col, task_name):
    """NLI has no training component — classify all samples directly."""
    print(f"\n[Zero-shot NLI] task={task_name}  (parameter-free, evaluated on full n={len(df)})")
    le = LabelEncoder().fit(df[label_col])
    class_names = list(le.classes_)
    maj = majority_baseline(df[label_col].tolist())
    hyp_map = GOAL_HYPOTHESES if label_col == "goal" else SAT_HYPOTHESES

    preds  = le.transform(_nli_predict(df["student_text"].tolist(), hyp_map, class_names))
    y_true = le.transform(df[label_col])
    acc = accuracy_score(y_true, preds)
    mf1 = f1_score(y_true, preds, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true, preds, labels=range(len(class_names)))
    print(f"\n{'─'*55}")
    print(f"  Zero-shot NLI (bart-large-mnli)  [full dataset, n={len(df)}]")
    print(f"  Accuracy:  {acc:.3f}  (majority baseline: {maj:.3f}, Δ={acc-maj:+.3f})")
    print(f"  Macro-F1:  {mf1:.3f}")
    print(classification_report(y_true, preds, target_names=class_names, zero_division=0))
    plot_confusion_matrix(cm, class_names, f"Zero-shot NLI — {task_name} (full dataset)",
                          os.path.join(RESULTS, f"{task_name}_nli_cm.png"))
    return {"accuracy": acc, "f1": mf1}


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(task_name, results: dict):
    print(f"\n{'═'*60}")
    print(f"  SUMMARY — {task_name}")
    print(f"{'─'*60}")
    print(f"  {'Model':<35} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"{'─'*60}")
    for name, r in results.items():
        print(f"  {name:<35} {r['accuracy']:>10.3f} {r['f1']:>10.3f}")
    print(f"{'═'*60}")


def save_summary_csv(task_name, results: dict):
    path = os.path.join(RESULTS, f"{task_name}_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "macro_f1"])
        for name, r in results.items():
            w.writerow([name, f"{r['accuracy']:.4f}", f"{r['f1']:.4f}"])
    print(f"  Summary saved: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run_task(task_name: str, label_col: str, skip_cnn: bool,
             splits_data: dict | None, force_loocv: bool, n_splits: int = 5,
             use_gpt4: bool = False, use_nli: bool = False, gpt4_model: str = "gpt-4o-mini"):
    results = {}

    if splits_data and not force_loocv:
        # ── Fixed 8:1:1 split ───────────────────────────────────────────────
        csv_path = (os.path.join(BASE, "data", "labeled.csv")
                    if label_col == "satisfaction"
                    else os.path.join(BASE, "data", "processed.csv"))
        df_full = _load_raw_for_splits(csv_path, label_col)
        df_train, df_val, df_test = get_split_dfs(df_full, splits_data)
        maj = _majority_acc_fixed(df_train, df_test, label_col)
        print(f"\n{'═'*60}")
        print(f"  Task: {task_name}  [fixed 8:1:1 split]")
        print(f"  train={len(df_train)}  val={len(df_val)}  test={len(df_test)}  "
              f"  train-majority baseline: {maj:.3f}")
        print(f"  Classes (train): {dict(Counter(df_train[label_col]))}")
        print(f"{'═'*60}")
        results["TF-IDF + LogReg (text+scores)"] = run_tfidf_fixed(
            df_train, df_val, df_test, label_col, task_name)
        results["Sentence-MLP (text+scores)"] = run_sentence_mlp_fixed(
            df_train, df_val, df_test, label_col, task_name)
        if not skip_cnn:
            results["TextCNN (text only)"] = run_textcnn_fixed(
                df_train, df_val, df_test, label_col, task_name)
        if use_nli:
            results["Zero-shot NLI"] = run_zeroshot_nli_fixed(
                df_train, df_val, df_test, label_col, task_name)
        if use_gpt4:
            results[f"GPT-4 few-shot ({gpt4_model})"] = run_gpt4_fixed(
                df_train, df_val, df_test, label_col, task_name, model=gpt4_model)
    else:
        # ── LOOCV fallback ──────────────────────────────────────────────────
        if label_col == "satisfaction":
            df = load_satisfaction_data()
        else:
            df = load_goal_data()
        results["TF-IDF + LogReg (text+scores)"] = run_tfidf_logreg(
            df, label_col, task_name, n_splits)
        results["Sentence-MLP (text+scores)"] = run_sentence_mlp(
            df, label_col, task_name, n_splits)
        if not skip_cnn:
            results["TextCNN (text only)"] = run_textcnn(df, label_col, task_name, n_splits)
        if use_nli:
            results["Zero-shot NLI"] = run_zeroshot_nli_loocv(df, label_col, task_name)
        if use_gpt4:
            results[f"GPT-4 few-shot ({gpt4_model})"] = run_gpt4_loocv(
                df, label_col, task_name, model=gpt4_model)

    print_summary(task_name, results)
    save_summary_csv(task_name, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["goal", "satisfaction", "both"], default="both")
    parser.add_argument("--no-cnn", action="store_true", help="Skip TextCNN (faster)")
    parser.add_argument("--loocv", action="store_true",
                        help="Force LOOCV even when splits.json exists")
    parser.add_argument("--folds", type=int, default=5,
                        help="K for stratified k-fold (LOOCV mode only, default 5)")
    parser.add_argument("--gpt4", action="store_true",
                        help="Add GPT-4 few-shot classifier (requires OPENAI_API_KEY)")
    parser.add_argument("--gpt4-model", default="gpt-4o-mini",
                        help="GPT-4 model variant (default: gpt-4o-mini)")
    parser.add_argument("--nli", action="store_true",
                        help="Add zero-shot NLI classifier (downloads ~1.6 GB model first run)")
    args = parser.parse_args()

    splits_data = load_splits()
    if splits_data and not args.loocv:
        print(f"\nUsing pre-computed splits from data/splits.json "
              f"(created {splits_data.get('created', '?')}, "
              f"seed={splits_data.get('seed')}, "
              f"min_class={splits_data.get('min_class_size')})")
    else:
        if args.loocv:
            print("\n--loocv: using Leave-One-Out Cross-Validation")
        else:
            print("\nNo splits.json found — falling back to LOOCV")

    if args.task in ("goal", "both"):
        run_task("goal", "goal", skip_cnn=args.no_cnn,
                 splits_data=splits_data, force_loocv=args.loocv, n_splits=args.folds,
                 use_gpt4=args.gpt4, use_nli=args.nli, gpt4_model=args.gpt4_model)

    if args.task in ("satisfaction", "both"):
        labeled_csv = os.path.join(BASE, "data", "labeled.csv")
        if not os.path.exists(labeled_csv):
            print("\nSkipping satisfaction task — labeled.csv not found.")
            print("Run: python scripts/llm_label.py")
        else:
            sat_splits = splits_data if (splits_data and splits_data.get("label_col") == "satisfaction") else None
            run_task("satisfaction", "satisfaction", skip_cnn=args.no_cnn,
                     splits_data=sat_splits, force_loocv=args.loocv, n_splits=args.folds,
                     use_gpt4=args.gpt4, use_nli=args.nli, gpt4_model=args.gpt4_model)


if __name__ == "__main__":
    main()
