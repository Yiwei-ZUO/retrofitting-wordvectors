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
from src.preprocessing import build_wolf_graph, filter_graph_by_vocab
from evaluation.eval_utils import load_fr_simlex, evaluate_with_coverage


MAX_WORDS = 300000
NUM_ITERS = 10

ALPHAS = [0.111, 0.250, 0.429, 0.667, 1.0, 1.5, 2.333, 4.0, 9.0]

FASTTEXT_PATH = PROJECT_ROOT / "models/cc.fr.300.vec"
WOLF_PATH     = PROJECT_ROOT / "datasets/wolf/wolf-1.0b4.xml"
FR_SIMLEX_PATH = PROJECT_ROOT / "datasets/fr-simlex.dataset"

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV  = RESULTS_DIR / "french_alpha_search_simlex_results.csv"
OUTPUT_PLOT = RESULTS_DIR / "french_alpha_search_simlex_curve.png"
OUTPUT_BEST = RESULTS_DIR / "best_alpha_french.txt"


def assert_all_finite(vectors, label):
    for word, vector in vectors.items():
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} has non-finite values for: {word}")


def plot_curve(rows, original_rho, best_alpha):
    x = [r["alpha"] for r in rows]
    y = [r["retrofitted_rho"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, marker="o", color="#111111", linewidth=2.5, label="French SimLex rho")
    ax.axhline(original_rho, color="#888888", linestyle="--", linewidth=1,
               label=f"Original fastText baseline ({original_rho:.4f})")
    ax.axvline(best_alpha, color="red", linestyle=":", linewidth=1.5,
               label=f"Best alpha = {best_alpha:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)", fontsize=12)
    ax.set_ylabel("Spearman rho on French SimLex", fontsize=12)
    ax.set_title("Alpha Search on French SimLex\nfastText + WOLF_syn, inverse_degree beta",
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

    log("=== Alpha Search on French SimLex ===")
    log("fastText + WOLF_syn, inverse_degree beta")
    log(f"Alpha values: {ALPHAS}")
    log()

    log(f"=== Loading French fastText (limited to {MAX_WORDS} words) ===")
    t0 = time.perf_counter()
    embeddings = load_text_embeddings(FASTTEXT_PATH, max_words=MAX_WORDS)
    log(f"Loaded: {len(embeddings)} words in {time.perf_counter()-t0:.1f}s")
    assert_all_finite(embeddings, "Original embeddings")
    log()

    log("=== Building WOLF_syn graph ===")
    t0 = time.perf_counter()
    graph = build_wolf_graph(WOLF_PATH, include_synonyms=True,
                             include_hypernyms=False, include_hyponyms=False,
                             keep_only_vocab=False)
    graph = filter_graph_by_vocab(graph, set(embeddings))
    log(f"Graph nodes: {len(graph)} built in {time.perf_counter()-t0:.1f}s")
    log()

    log("=== Loading French SimLex ===")
    pairs = load_fr_simlex(FR_SIMLEX_PATH)

    log("=== Evaluating original fastText baseline ===")
    original_result = evaluate_with_coverage(embeddings, pairs)
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

        assert_all_finite(retrofitted, f"Retrofitted alpha={alpha}")
        log(f"  Runtime: {runtime:.2f}s  |  stats: {stats}")

        after = evaluate_with_coverage(retrofitted, pairs)
        delta = after["rho"] - original_result["rho"]

        rows.append({
            "alpha": alpha,
            "dataset": "French_SimLex",
            "original_rho": round(original_result["rho"], 4),
            "retrofitted_rho": round(after["rho"], 4),
            "delta": round(delta, 4),
            "covered_pairs": after["covered"],
            "total_pairs": after["total"],
            "runtime_seconds": round(runtime, 2),
        })
        log(f"  French_SimLex: {original_result['rho']:.4f} -> {after['rho']:.4f} ({delta:+.4f})")
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
    log(f"=== BEST ALPHA (French) = {best_alpha:.3f} "
        f"(French_SimLex rho={best_row['retrofitted_rho']:.4f}) ===")

    plot_curve(rows, original_result["rho"], best_alpha)

    with open(OUTPUT_BEST, "w", encoding="utf-8") as f:
        f.write(str(best_alpha))
    log(f"Best French alpha saved to {OUTPUT_BEST}")

    log_path = RESULTS_DIR / "french_alpha_search_summary.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Log saved to {log_path}")
    log("=== Done ===")


if __name__ == "__main__":
    main()
