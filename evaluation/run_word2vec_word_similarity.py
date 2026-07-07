from pathlib import Path
import csv
import sys
import time
import gc

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_word2vec_binary
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab
from evaluation.eval_utils import load_ws353, load_rg65, evaluate_with_coverage


BEST_ALPHA = 0.429
NUM_ITERS = 10
MAX_WORDS = 300000  # limit to reduce memory pressure

WORD2VEC_PATH = PROJECT_ROOT / "models/GoogleNews-vectors-negative300.bin.gz"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV  = RESULTS_DIR / "word2vec_word_similarity_results.csv"
OUTPUT_PLOT = RESULTS_DIR / "word2vec_word_similarity_bar.png"


def assert_all_finite(vectors, label):
    for word, vector in vectors.items():
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} has non-finite values for: {word}")


def run_combo(name, embeddings, graph, datasets):
    print(f"\n=== Combo: {name} ===")
    print(f"Embeddings: {len(embeddings)}  |  Graph nodes: {len(graph)}")

    t0 = time.perf_counter()
    retrofitted, stats = retrofit_vectors(
        embeddings, graph, num_iters=NUM_ITERS, alpha=BEST_ALPHA,
    )
    runtime = time.perf_counter() - t0
    assert_all_finite(retrofitted, name)
    print(f"Retrofitting done in {runtime:.1f}s. Stats: {stats}")

    rows = []
    for ds_name, pairs in datasets.items():
        before = evaluate_with_coverage(embeddings, pairs)
        after  = evaluate_with_coverage(retrofitted, pairs)
        delta  = after["rho"] - before["rho"]
        rows.append({
            "combo": name, "dataset": ds_name,
            "original_rho": round(before["rho"], 4),
            "retrofitted_rho": round(after["rho"], 4),
            "delta": round(delta, 4),
            "covered_pairs": after["covered"],
            "total_pairs": after["total"],
        })
        print(f"  [{ds_name}] {before['rho']:.4f} -> {after['rho']:.4f} ({delta:+.4f})")

    del retrofitted
    gc.collect()
    return rows


def plot_results(all_rows, save_path):
    combos = sorted(set(r["combo"] for r in all_rows))
    datasets = sorted(set(r["dataset"] for r in all_rows))

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        ds_rows = {r["combo"]: r for r in all_rows if r["dataset"] == ds_name}
        original_rho = ds_rows[combos[0]]["original_rho"]

        labels = ["Original"] + combos
        values = [original_rho] + [ds_rows[c]["retrofitted_rho"] for c in combos]
        colors = ["#888888"] + ["#111111"] * len(combos)

        bars = ax.barh(labels, values, color=colors)
        for bar in bars:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.3f}", va="center", fontsize=9)
        ax.set_xlabel("Spearman rho")
        ax.set_title(ds_name)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(f"Word2Vec Word Similarity: 2 Lexicon Combos (alpha={BEST_ALPHA})",
                fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== Loading evaluation datasets ===")
    datasets = {
        "WS-353": load_ws353(PROJECT_ROOT / "datasets/en-ws353.csv"),
        "RG-65":  load_rg65(PROJECT_ROOT  / "datasets/en_rg65.txt"),
    }

    print(f"\nLoading Word2Vec (limited to {MAX_WORDS} most frequent words)...")
    w2v = load_word2vec_binary(WORD2VEC_PATH, max_words=MAX_WORDS)
    print(f"Loaded {len(w2v)} words")

    all_rows = []

    # Combo: Word2Vec + WN_syn
    graph_syn = build_wordnet_graph(set(w2v), include_synonyms=True,
                                    include_hypernyms=False, include_hyponyms=False)
    graph_syn = filter_graph_by_vocab(graph_syn, set(w2v))
    all_rows += run_combo("Word2Vec+WN_syn", w2v, graph_syn, datasets)
    del graph_syn
    gc.collect()

    # Combo: Word2Vec + WN_all
    graph_all = build_wordnet_graph(set(w2v), include_synonyms=True,
                                    include_hypernyms=True, include_hyponyms=True)
    graph_all = filter_graph_by_vocab(graph_all, set(w2v))
    all_rows += run_combo("Word2Vec+WN_all", w2v, graph_all, datasets)
    del graph_all
    gc.collect()

    fieldnames = ["combo", "dataset", "original_rho", "retrofitted_rho",
                  "delta", "covered_pairs", "total_pairs"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved to {OUTPUT_CSV}")

    plot_results(all_rows, OUTPUT_PLOT)
    print("\n=== Word2Vec combos done ===")


if __name__ == "__main__":
    main()
