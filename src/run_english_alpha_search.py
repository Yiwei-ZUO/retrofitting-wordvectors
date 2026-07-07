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
from src.utils import load_text_embeddings
from src.preprocessing import build_wordnet_graph, filter_graph_by_vocab
from evaluation.eval_utils import load_simlex, evaluate_with_coverage


MAX_WORDS = None  # full GloVe vocabulary
NUM_ITERS = 10

# desired weight 0.1 -> alpha 0.111 ... 0.9 -> alpha 9.0
ALPHAS = [0.111, 0.250, 0.429, 0.667, 1.0, 1.5, 2.333, 4.0, 9.0]

EMBEDDING_PATH = PROJECT_ROOT / "models/glove.6B.300d.txt"
RESULTS_DIR    = PROJECT_ROOT / "results"
OUTPUT_CSV     = RESULTS_DIR / "english_alpha_search_simlex_results.csv"
OUTPUT_LOG     = RESULTS_DIR / "english_alpha_search_simlex_summary.txt"
OUTPUT_PLOT    = RESULTS_DIR / "english_alpha_search_simlex_curve.png"
OUTPUT_BEST    = RESULTS_DIR / "best_alpha.txt"


def assert_same_vocabulary(original, retrofitted):
    if set(original) != set(retrofitted):
        raise AssertionError("Vocabulary mismatch after retrofitting.")


def assert_all_finite(vectors, label):
    for word, vector in vectors.items():
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} has non-finite values for: {word}")


def plot_curve(rows, original_rho, best_alpha):
    x = [r["alpha"] for r in rows]
    y = [r["retrofitted_rho"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, marker="o", color="#111111", linewidth=2.5, label="SimLex-999 rho")
    ax.axhline(original_rho, color="#888888", linestyle="--", linewidth=1,
               label=f"Original GloVe baseline ({original_rho:.4f})")
    ax.axvline(best_alpha, color="red", linestyle=":", linewidth=1.5,
               label=f"Best alpha = {best_alpha:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)  -  higher = trust original vector more", fontsize=12)
    ax.set_ylabel("Spearman rho on SimLex-999", fontsize=12)
    ax.set_title("Alpha Search on SimLex-999\nWN_syn + inverse_degree beta, full GloVe vocabulary",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#DDDDDD")
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()
    print(f"Plot saved to {OUTPUT_PLOT}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("=== Alpha Search on SimLex-999 only ===")
    log("WN_syn + inverse_degree beta, FULL GloVe vocabulary")
    log(f"Alpha values: {ALPHAS}")
    log()

    log("=== Loading GloVe embeddings (full vocabulary) ===")
    t0 = time.perf_counter()
    embeddings = load_text_embeddings(EMBEDDING_PATH, max_words=MAX_WORDS)
    log(f"Loaded: {len(embeddings)} words in {time.perf_counter()-t0:.1f}s")
    assert_all_finite(embeddings, "Original embeddings")
    log()

    log("=== Building WN_syn graph ===")
    t0 = time.perf_counter()
    graph = build_wordnet_graph(set(embeddings), include_synonyms=True,
                                include_hypernyms=False, include_hyponyms=False)
    graph = filter_graph_by_vocab(graph, set(embeddings))
    log(f"Graph nodes: {len(graph)} built in {time.perf_counter()-t0:.1f}s")
    log()

    log("=== Loading SimLex-999 ===")
    simlex = load_simlex(PROJECT_ROOT / "datasets/en-simlex.txt")

    log("=== Evaluating original GloVe baseline on SimLex-999 ===")
    original_result = evaluate_with_coverage(embeddings, simlex)
    log(f"Original rho={original_result['rho']:.4f}, "
        f"coverage={original_result['covered']}/{original_result['total']}")
    log()

    rows = []

    for alpha in ALPHAS:
        log(f"=== alpha={alpha:.3f} ===")
        start = time.perf_counter()

        retrofitted, stats = retrofit_vectors(
            embeddings, graph, num_iters=NUM_ITERS, alpha=alpha,
        )
        runtime = time.perf_counter() - start

        assert_same_vocabulary(embeddings, retrofitted)
        assert_all_finite(retrofitted, f"Retrofitted alpha={alpha}")
        log(f"  Runtime: {runtime:.2f}s  |  stats: {stats}")

        after = evaluate_with_coverage(retrofitted, simlex)
        delta = after["rho"] - original_result["rho"]

        rows.append({
            "alpha": alpha,
            "dataset": "SimLex-999",
            "original_rho": round(original_result["rho"], 4),
            "retrofitted_rho": round(after["rho"], 4),
            "delta": round(delta, 4),
            "covered_pairs": after["covered"],
            "total_pairs": after["total"],
            "runtime_seconds": round(runtime, 2),
        })
        log(f"  SimLex-999: {original_result['rho']:.4f} -> {after['rho']:.4f} ({delta:+.4f})")
        log()

        del retrofitted
        gc.collect()

    fieldnames = ["alpha", "dataset", "original_rho",
                  "retrofitted_rho", "delta", "covered_pairs", "total_pairs",
                  "runtime_seconds"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"CSV saved to {OUTPUT_CSV}")

    best_row = max(rows, key=lambda r: r["retrofitted_rho"])
    best_alpha = best_row["alpha"]
    log()
    log(f"=== BEST ALPHA = {best_alpha:.3f} (SimLex-999 rho={best_row['retrofitted_rho']:.4f}) ===")

    plot_curve(rows, original_result["rho"], best_alpha)

    with open(OUTPUT_BEST, "w", encoding="utf-8") as f:
        f.write(str(best_alpha))
    log(f"Best alpha saved to {OUTPUT_BEST}")

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Log saved to {OUTPUT_LOG}")
    log("=== Done ===")


if __name__ == "__main__":
    main()
