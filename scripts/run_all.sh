#!/usr/bin/env bash
# End-to-end pipeline runner for the Student–LLM Tutoring project.
#
# Usage:
#   ./scripts/run_all.sh                    # full pipeline
#   OPENAI_API_KEY=sk-... ./scripts/run_all.sh
#   ./scripts/run_all.sh --skip-labeling    # skip LLM labeling (goal task only)
#   ./scripts/run_all.sh --no-cnn           # skip TextCNN
#   ./scripts/run_all.sh --gpt4             # include GPT-4 few-shot classifier
#   ./scripts/run_all.sh --nli              # include zero-shot NLI classifier

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$ROOT/venv/bin/python"

SKIP_LABELING=0
NO_CNN=""
GPT4=""
NLI=""

for arg in "$@"; do
  case $arg in
    --skip-labeling) SKIP_LABELING=1 ;;
    --no-cnn)        NO_CNN="--no-cnn" ;;
    --gpt4)          GPT4="--gpt4" ;;
    --nli)           NLI="--nli" ;;
  esac
done

echo "════════════════════════════════════════"
echo "Step 1: Repair malformed JSON files"
echo "════════════════════════════════════════"
"$PYTHON" "$SCRIPT_DIR/repair_json.py"

echo ""
echo "════════════════════════════════════════"
echo "Step 2: Preprocess dialogues → processed.csv"
echo "════════════════════════════════════════"
"$PYTHON" "$SCRIPT_DIR/preprocess.py"

echo ""
echo "════════════════════════════════════════"
echo "Step 3: Build 8:1:1 train/val/test split → splits.json"
echo "════════════════════════════════════════"
"$PYTHON" "$SCRIPT_DIR/split_data.py"

if [ "$SKIP_LABELING" -eq 0 ]; then
  echo ""
  echo "════════════════════════════════════════"
  echo "Step 4: LLM labeling (OpenAI) → labeled.csv"
  echo "════════════════════════════════════════"
  if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set — skipping labeling."
    echo "  Export OPENAI_API_KEY=sk-... and re-run, or use --skip-labeling."
    SKIP_LABELING=1
  else
    "$PYTHON" "$SCRIPT_DIR/llm_label.py" --model gpt-4o-mini
  fi
fi

echo ""
echo "════════════════════════════════════════"
echo "Step 5: Train classifiers"
echo "════════════════════════════════════════"
if [ "$SKIP_LABELING" -eq 1 ]; then
  "$PYTHON" "$SCRIPT_DIR/train_classifiers.py" --task goal $NO_CNN $GPT4 $NLI
else
  "$PYTHON" "$SCRIPT_DIR/train_classifiers.py" --task both $NO_CNN $GPT4 $NLI
fi

echo ""
echo "════════════════════════════════════════"
echo "Done. Results are in: $ROOT/results/"
echo "════════════════════════════════════════"
