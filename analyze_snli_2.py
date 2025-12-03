#!/usr/bin/env python
"""
SNLI + ELECTRA Analysis Script

Follows the CS388 final project spec:
- Overall accuracy
- Confusion matrix
- Slice analysis (negation, lexical overlap)
- Competency problems (biased n-grams)
- Saves tables (CSV) and plots (PNG)
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------
# Loading and helpers
# --------------------

def load_eval_predictions(path):
    """
    Load eval_predictions.jsonl produced by run.py

    Each line is expected to have at least:
      - "premise"
      - "hypothesis"
      - "label" (gold label id: 0,1,2)
      - "predicted_label" (predicted label id)
      - "predicted_scores" (optional: list of logits)
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def compute_accuracy(examples):
    if not examples:
        return 0.0
    correct = sum(1 for ex in examples if ex["label"] == ex["predicted_label"])
    return correct / len(examples)


def contains_negation(text: str) -> bool:
    text_l = text.lower()
    tokens = text_l.split()
    neg_words = ["not", "no", "never", "nobody", "nothing", "none"]
    if any(w in tokens for w in neg_words):
        return True
    if "n't" in text_l:
        return True
    return False


def lexical_overlap(premise: str, hypothesis: str) -> float:
    """
    Fraction of hypothesis tokens that also appear in the premise.
    """
    p = set(premise.lower().split())
    h = set(hypothesis.lower().split())
    if not h:
        return 0.0
    return len(p & h) / len(h)


def confusion_matrix(examples, num_labels=3):
    """
    Returns a num_labels x num_labels matrix of counts.
    Rows: gold, Columns: predicted.
    """
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for ex in examples:
        g = ex["label"]
        p = ex["predicted_label"]
        if 0 <= g < num_labels and 0 <= p < num_labels:
            cm[g, p] += 1
    return cm


# --------------------
# N-gram analysis (competency problems)
# --------------------

def top_ngrams(examples, n=1, top_k=20, min_count=5):
    """
    Find n-grams in the hypothesis that are strongly biased toward a single label.

    Returns:
        List of tuples (ngram, bias_score, total_count, counts_per_label_dict)
    """
    counts = defaultdict(lambda: Counter())

    for ex in examples:
        label = ex["label"]
        words = ex["hypothesis"].lower().split()
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            counts[gram][label] += 1

    scored = []
    for gram, c in counts.items():
        total = sum(c.values())
        if total < min_count:
            continue
        best_label, best_count = c.most_common(1)[0]
        bias_score = best_count / total
        scored.append((gram, bias_score, total, dict(c)))

    # sort by (bias_score, total_count) descending
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored[:top_k]


# --------------------
# Plotting helpers
# --------------------

def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    plt.colorbar(im, ax=ax)
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_bar(xs, ys, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(xs, ys)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# --------------------
# Main analysis
# --------------------

def analyze(eval_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    examples = load_eval_predictions(eval_path)
    print(f"Loaded {len(examples)} eval examples from {eval_path}")

    # ---- Overall accuracy ----
    overall_acc = compute_accuracy(examples)
    print("\n=== Overall accuracy ===")
    print(f"Accuracy: {overall_acc * 100:.2f}%")

    # ---- Confusion matrix ----
    cm = confusion_matrix(examples, num_labels=3)
    labels = ["entailment(0)", "neutral(1)", "contradiction(2)"]
    print("\n=== Confusion Matrix (gold rows × predicted columns) ===")
    print(cm)
    # save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    # plot heatmap
    plot_confusion_matrix(
        cm,
        ["Entailment", "Neutral", "Contradiction"],
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    # ---- Negation slice ----
    neg_examples = [ex for ex in examples if contains_negation(ex["hypothesis"])]
    nonneg_examples = [ex for ex in examples if not contains_negation(ex["hypothesis"])]

    neg_acc = compute_accuracy(neg_examples)
    nonneg_acc = compute_accuracy(nonneg_examples)

    print("\n=== Negation slice (hypothesis contains negation cue) ===")
    print(f"Negation examples: {len(neg_examples)}, accuracy: {neg_acc * 100:.2f}%")
    print(f"Non-negation examples: {len(nonneg_examples)}, accuracy: {nonneg_acc * 100:.2f}%")

    neg_slice_df = pd.DataFrame(
        [
            {"slice": "all", "count": len(examples), "accuracy": overall_acc},
            {"slice": "negation", "count": len(neg_examples), "accuracy": neg_acc},
            {"slice": "non_negation", "count": len(nonneg_examples), "accuracy": nonneg_acc},
        ]
    )
    neg_slice_df.to_csv(os.path.join(output_dir, "negation_slice.csv"), index=False)

    plot_bar(
        ["All", "Negation", "Non-negation"],
        [overall_acc * 100, neg_acc * 100, nonneg_acc * 100],
        ylabel="Accuracy (%)",
        title="Accuracy by negation slice",
        save_path=os.path.join(output_dir, "negation_slice.png"),
    )

    # ---- Lexical overlap buckets ----
    # Lexical overlap measures how many words in the hypothesis also appear in the premise.
    overlaps = [lexical_overlap(ex["premise"], ex["hypothesis"]) for ex in examples]
    low, med, high = [], [], []
    for ex, o in zip(examples, overlaps):
        if o < 0.2:
            low.append(ex)
        elif o < 0.5:
            med.append(ex)
        else:
            high.append(ex)

    low_acc = compute_accuracy(low)
    med_acc = compute_accuracy(med)
    high_acc = compute_accuracy(high)

    print("\n=== Lexical overlap buckets (premise–hypothesis) ===")
    print(f"Low (<0.2): count={len(low)}, acc={low_acc * 100:.2f}%")
    print(f"Medium (0.2–0.5): count={len(med)}, acc={med_acc * 100:.2f}%")
    print(f"High (>=0.5): count={len(high)}, acc={high_acc * 100:.2f}%")

    overlap_df = pd.DataFrame(
        [
            {"bucket": "low", "count": len(low), "accuracy": low_acc},
            {"bucket": "medium", "count": len(med), "accuracy": med_acc},
            {"bucket": "high", "count": len(high), "accuracy": high_acc},
        ]
    )
    overlap_df.to_csv(os.path.join(output_dir, "overlap_buckets.csv"), index=False)

    plot_bar(
        ["Low", "Medium", "High"],
        [low_acc * 100, med_acc * 100, high_acc * 100],
        ylabel="Accuracy (%)",
        title="Accuracy by lexical overlap",
        save_path=os.path.join(output_dir, "overlap_buckets.png"),
    )

    # ---- Top biased n-grams (1-grams) ----
    top_1grams = top_ngrams(examples, n=1, top_k=20, min_count=5)
    print("\n=== Top 1-grams most strongly correlated with a label ===")
    ngram_rows = []
    for gram, bias_score, total, counts in top_1grams:
        print(f"{gram:15s} bias_score={bias_score:.2f} total={total} counts={counts}")
        majority_label = max(counts.items(), key=lambda kv: kv[1])[0]
        ngram_rows.append(
            {
                "ngram": gram,
                "total": total,
                "bias_score": bias_score,
                "majority_label": majority_label,
                "counts": counts,
            }
        )

    ngram_df = pd.DataFrame(ngram_rows)
    ngram_df.to_csv(os.path.join(output_dir, "top_1grams.csv"), index=False)

    # DONE
    print(f"\nAnalysis complete. Tables and plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_predictions",
        type=str,
        required=True,
        help="Path to eval_predictions.jsonl produced by run.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis outputs (CSVs and PNGs).",
    )
    args = parser.parse_args()
    analyze(args.eval_predictions, args.output_dir)


if __name__ == "__main__":
    main()
