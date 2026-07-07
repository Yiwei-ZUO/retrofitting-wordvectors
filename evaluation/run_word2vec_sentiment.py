from pathlib import Path
import csv
import sys
import time
import gc

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_word2vec_binary
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab


BEST_ALPHA = 0.429
NUM_ITERS = 10
MAX_WORDS = 300000

WORD2VEC_PATH = PROJECT_ROOT / "models/GoogleNews-vectors-negative300.bin.gz"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV  = RESULTS_DIR / "word2vec_sentiment_results.csv"
OUTPUT_PLOT = RESULTS_DIR / "word2vec_sentiment_bar.png"


def sentence_to_vector(sentence, vectors):
    words = sentence.lower().split()
    vecs = [vectors[w] for w in words if w in vectors]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def evaluate_sentiment(vectors, ds, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    def encode(split):
        X, y = [], []
        for ex in ds[split]:
            if ex["label"] == -1:
                continue
            vec = sentence_to_vector(ex["sentence"], vectors)
            if vec is not None:
                X.append(vec)
                y.append(ex["label"])
        return np.array(X), np.array(y)

    X_train, y_train = encode("train")
    X_val, y_val = encode("validation")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"  [{label}] accuracy: {acc:.4f}")
    return acc


def run_combo(name, embeddings, graph, ds):
    print(f"\n=== Combo: {name} ===")
    print(f"Embeddings: {len(embeddings)}  |  Graph nodes: {len(graph)}")

    t0 = time.perf_counter()
    retrofitted, stats = retrofit_vectors(
        embeddings, graph, num_iters=NUM_ITERS, alpha=BEST_ALPHA,
    )
    print(f"Retrofitting done in {time.perf_counter()-t0:.1f}s")

    acc = evaluate_sentiment(retrofitted, ds, name)

    del retrofitted
    gc.collect()
    return acc


def plot_results(rows, original_acc, save_path):
    labels = ["Original"] + [r["combo"] for r in rows]
    values = [original_acc] + [r["accuracy"] for r in rows]
    colors = ["#888888"] + ["#111111"] * len(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(labels, values, color=colors)
    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=10)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0.5, 0.85)
    ax.set_title(f"Word2Vec Sentiment (SST-2): 2 Lexicon Combos (alpha={BEST_ALPHA})",
                fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_path}")


def main():
    from datasets import load_dataset

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SST-2...")
    ds = load_dataset("stanfordnlp/sst2")

    print(f"\nLoading Word2Vec (limited to {MAX_WORDS} most frequent words)...")
    w2v = load_word2vec_binary(WORD2VEC_PATH, max_words=MAX_WORDS)
    print(f"Loaded {len(w2v)} words")

    print("\nEvaluating Original Word2Vec...")
    original_acc = evaluate_sentiment(w2v, ds, "Original Word2Vec")

    rows = []

    # Combo: Word2Vec + WN_syn
    graph_syn = build_wordnet_graph(set(w2v), include_synonyms=True,
                                    include_hypernyms=False, include_hyponyms=False)
    graph_syn = filter_graph_by_vocab(graph_syn, set(w2v))
    acc = run_combo("Word2Vec+WN_syn", w2v, graph_syn, ds)
    rows.append({"combo": "Word2Vec+WN_syn", "accuracy": round(acc, 4),
                 "original_accuracy": round(original_acc, 4), "delta": round(acc - original_acc, 4)})
    del graph_syn
    gc.collect()

    # Combo: Word2Vec + WN_all
    graph_all = build_wordnet_graph(set(w2v), include_synonyms=True,
                                    include_hypernyms=True, include_hyponyms=True)
    graph_all = filter_graph_by_vocab(graph_all, set(w2v))
    acc = run_combo("Word2Vec+WN_all", w2v, graph_all, ds)
    rows.append({"combo": "Word2Vec+WN_all", "accuracy": round(acc, 4),
                 "original_accuracy": round(original_acc, 4), "delta": round(acc - original_acc, 4)})
    del graph_all
    gc.collect()

    fieldnames = ["combo", "accuracy", "original_accuracy", "delta"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {OUTPUT_CSV}")

    plot_results(rows, original_acc, OUTPUT_PLOT)
    print("\n=== Word2Vec sentiment combos done ===")


if __name__ == "__main__":
    main()
