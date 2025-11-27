import json
from collections import Counter
from pathlib import Path

# ---------- Helpers ----------

def load_eval_predictions(path: str):
    path = Path(path)
    examples = []
    with path.open() as f:
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
    neg_words = ["not", "no", "never", "nobody", "nothing", "none"]
    # crude but fine: check token-level and "n't" as substring
    tokens = text_l.split()
    if any(w in tokens for w in neg_words):
        return True
    if "n't" in text_l:
        return True
    return False


def hypothesis_length(text: str) -> int:
    return len(text.split())

def word_overlap(premise, hypothesis):
    p = set(premise.lower().split())
    h = set(hypothesis.lower().split())
    if len(h) == 0:
        return 0
    return len(p & h) / len(h)  # fraction of hypothesis words also in premise

def confusion_matrix(examples):
    cm = [[0,0,0],[0,0,0],[0,0,0]]  # gold x pred
    for ex in examples:
        g = ex["label"]
        p = ex["predicted_label"]
        cm[g][p] += 1
    return cm

from collections import defaultdict, Counter

def top_ngrams(examples, n=1, top_k=20, min_count=5):
    """
    Find n-grams in the hypothesis that are strongly biased toward a single label.

    Args:
        examples: list of dicts with fields including "hypothesis" and "label"
        n: n-gram length (1 = unigram, 2 = bigram, etc.)
        top_k: how many top n-grams to return
        min_count: minimum number of occurrences required to consider an n-gram
    """
    # counts[gram][label] = how many times this n-gram appears with this label
    counts = defaultdict(lambda: Counter())

    # ----- 1. Collect counts for each n-gram -----
    for ex in examples:
        label = ex["label"]
        words = ex["hypothesis"].lower().split()
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i+n])  # n words joined into one string
            counts[gram][label] += 1

    # ----- 2. Compute bias score for each n-gram -----
    scored = []
    for gram, c in counts.items():
        total = sum(c.values())  # total occurrences of this n-gram

        # skip rare n-grams; they look biased just by chance
        if total < min_count:
            continue

        # best_label = (label_id, count_of_that_label)
        best_label = c.most_common(1)[0]
        bias_score = best_label[1] / total  # fraction of times it appears with that label

        scored.append((gram, bias_score, total, dict(c)))

    # ----- 3. Sort by strongest skew, break ties by frequency -----
    # First sort by bias_score (descending), then by total count (descending)
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Return top_k entries: (ngram, bias_score, total_count, counts_per_label)
    return scored[:top_k]



# ---------- Main analysis ----------

def analyze(eval_predictions_path: str):
    examples = load_eval_predictions(eval_predictions_path)
    print(f"Loaded {len(examples)} eval examples")

    # Overall accuracy
    overall_acc = compute_accuracy(examples)
    print(f"\n=== Overall ===")
    print(f"Accuracy: {overall_acc:.3f}")

    # Negation vs non-negation
    neg_examples = [ex for ex in examples if contains_negation(ex["hypothesis"])]
    nonneg_examples = [ex for ex in examples if not contains_negation(ex["hypothesis"])]

    neg_acc = compute_accuracy(neg_examples)
    nonneg_acc = compute_accuracy(nonneg_examples)

    print(f"\n=== Negation slice (hypothesis contains 'not'/negation) ===")
    print(f"Count: {len(neg_examples)}")
    print(f"Accuracy: {neg_acc:.3f}")

    print(f"\n=== Non-negation slice ===")
    print(f"Count: {len(nonneg_examples)}")
    print(f"Accuracy: {nonneg_acc:.3f}")

    # Length buckets
    buckets = {
        "short (<=5)": [],
        "medium (6-10)": [],
        "long (>10)": [],
    }

    for ex in examples:
        L = hypothesis_length(ex["hypothesis"])
        if L <= 5:
            buckets["short (<=5)"].append(ex)
        elif L <= 10:
            buckets["medium (6-10)"].append(ex)
        else:
            buckets["long (>10)"].append(ex)

    print(f"\n=== Hypothesis length buckets ===")
    for name, exs in buckets.items():
        acc = compute_accuracy(exs)
        print(f"{name}: count={len(exs)}, acc={acc:.3f}")

    # Show a few illustrative errors for your writeup
    print("\n=== Sample errors on negation examples ===")
    shown = 0
    for ex in neg_examples:
        if ex["label"] != ex["predicted_label"]:
            print("-" * 40)
            print(f"Premise   : {ex['premise']}")
            print(f"Hypothesis: {ex['hypothesis']}")
            print(f"Gold label: {ex['label']}")
            print(f"Pred label: {ex['predicted_label']}")
            shown += 1
            if shown >= 5:
                break

    # Word-overlap analysis
    overlaps = [word_overlap(ex["premise"], ex["hypothesis"]) for ex in examples]

    high_overlap = [ex for ex, o in zip(examples, overlaps) if o >= 0.5]
    mid_overlap  = [ex for ex, o in zip(examples, overlaps) if 0.2 <= o < 0.5]
    low_overlap  = [ex for ex, o in zip(examples, overlaps) if o < 0.2]

    print("\n=== Word Overlap Slices (premise–hypothesis lexical overlap) ===")
    for name, exs in [
        ("High (>= 0.5)", high_overlap),
        ("Medium (0.2–0.5)", mid_overlap),
        ("Low (<0.2)", low_overlap),
    ]:
        print(f"{name}: count={len(exs)}, acc={compute_accuracy(exs):.3f}")

    print("\n=== Confusion Matrix (gold rows × predicted columns) ===")
    cm = confusion_matrix(examples)
    labels = ["entailment(0)", "neutral(1)", "contradiction(2)"]
    for i, row in enumerate(cm):
        print(f"{labels[i]}: {row}")

    print("\n=== Top 1-grams most strongly correlated with a label ===")
    for gram, score, total, counts in top_ngrams(examples, n=1, top_k=15, min_count=5):
        print(f"{gram:15s}  bias_score={score:.2f}  total={total}  counts={counts}")



if __name__ == "__main__":
    # change this to your actual output_dir
    eval_path = "/content/drive/MyDrive/MS-work/NLP-project/eval_predictions.jsonl"
    analyze(eval_path)
