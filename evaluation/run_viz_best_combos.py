from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrofit import retrofit_vectors
from src.utils import load_text_embeddings
from src.preprocessing import build_wordnet_graph, build_wolf_graph, filter_graph_by_vocab


# English word groups
EN_WORD_GROUPS = {
    "positive emotion": ["happy", "joyful", "glad", "cheerful"],
    "negative emotion": ["sad", "unhappy", "miserable"],
    "intelligence":     ["smart", "intelligent", "clever"],
    "animals":          ["dog", "cat", "animal"],
    "movement":         ["run", "walk", "jog", "sprint"],
    "size":             ["big", "large", "huge", "small"],
    "speed":            ["fast", "rapid", "quick", "slow"],
}

# French word groups (translated equivalents, must exist in fastText vocab)
FR_WORD_GROUPS = {
    "émotion positive": ["heureux", "joyeux", "content"],
    "intelligence":      ["intelligent", "futé", "malin"],
    "mouvement":         ["courir", "marcher", "sprint"],
    "taille":            ["grand", "gros", "énorme", "petit"],
    "vitesse":           ["rapide", "vite"],
}

GROUP_COLORS_EN = {
    "positive emotion": "#E63946",
    "negative emotion": "#457B9D",
    "intelligence":     "#2A9D8F",
    "animals":          "#E9C46A",
    "movement":         "#9B5DE5",
    "size":             "#F77F00",
    "speed":            "#06D6A0",
}

GROUP_COLORS_FR = {
    "émotion positive": "#E63946",
    "émotion négative": "#457B9D",
    "intelligence":      "#2A9D8F",
    "animaux":           "#E9C46A",
    "mouvement":         "#9B5DE5",
    "taille":            "#F77F00",
    "vitesse":           "#06D6A0",
}


def make_viz(embeddings, retrofitted, word_groups, group_colors,
            title, save_path):
    all_words = []
    word_to_group = {}
    for group, words in word_groups.items():
        for w in words:
            if w in embeddings and w in retrofitted:
                all_words.append(w)
                word_to_group[w] = group

    print(f"Visualizing {len(all_words)} words: {all_words}")

    orig_vecs  = np.array([embeddings[w]  for w in all_words])
    retro_vecs = np.array([retrofitted[w] for w in all_words])

    combined = np.vstack([orig_vecs, retro_vecs])
    pca = PCA(n_components=2)
    pca.fit(combined)

    orig_2d  = pca.transform(orig_vecs)
    retro_2d = pca.transform(retro_vecs)

    x_range = orig_2d[:, 0].max() - orig_2d[:, 0].min()
    y_range = orig_2d[:, 1].max() - orig_2d[:, 1].min()
    offset_scale = max(x_range, y_range) * 0.06  # increased from 0.035

    fig, ax = plt.subplots(figsize=(20, 15))  # larger canvas
    ax.set_facecolor("white")
    ax.grid(color="#EEEEEE", linewidth=0.8)

    placed_labels = []

    for i, word in enumerate(all_words):
        group = word_to_group[word]
        color = group_colors[group]
        x0, y0 = orig_2d[i]
        x1, y1 = retro_2d[i]

        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85))
        ax.scatter(x0, y0, color=color, s=55, zorder=5, edgecolors="white", linewidths=0.8)

        # 16 candidate directions in a full circle, increasing radius if needed
        best_pos = None
        for radius_mult in [1.0, 1.6, 2.2, 3.0]:
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            candidates = [
                (x0 + radius_mult * offset_scale * np.cos(a),
                 y0 + radius_mult * offset_scale * np.sin(a))
                for a in angles
            ]
            for cand_x, cand_y in candidates:
                collision = any(
                    abs(cand_x - px) < offset_scale * 0.9 and abs(cand_y - py) < offset_scale * 0.55
                    for px, py in placed_labels
                )
                if not collision:
                    best_pos = (cand_x, cand_y)
                    break
            if best_pos is not None:
                break
        if best_pos is None:
            best_pos = (x0 + offset_scale * 3, y0 + offset_scale * 3)

        placed_labels.append(best_pos)
        ax.annotate(word, xy=(x0, y0), xytext=best_pos,
                    fontsize=9, color=color, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.5, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.4))

    legend_patches = [
        mpatches.Patch(color=group_colors[g], label=g)
        for g in word_groups
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=11,
              framealpha=0.95)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)

    Path("results").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    # ---- English: GloVe + WN_all, alpha=0.429 (best RG-65 result) ----
    print("\n=== ENGLISH: GloVe + WN_all (alpha=0.429) ===")
    glove = load_text_embeddings(
        PROJECT_ROOT / "models/glove.6B.300d.txt", max_words=50000
    )
    graph_en = build_wordnet_graph(set(glove), include_synonyms=True,
                                   include_hypernyms=True, include_hyponyms=True)
    graph_en = filter_graph_by_vocab(graph_en, set(glove))
    retrofitted_en, _ = retrofit_vectors(glove, graph_en, num_iters=10, alpha=0.429)

    make_viz(
        glove, retrofitted_en, EN_WORD_GROUPS, GROUP_COLORS_EN,
        "Word Vector Movement After Retrofitting (PCA 2D)\n"
        "English: GloVe + WordNet (syn+hyper+hypo), alpha=0.429",
        "results/eval_viz_english_best.png"
    )

    # ---- French: fastText + WOLF_all, alpha=2.333 (best RG-65_FR result) ----
    print("\n=== FRENCH: fastText + WOLF_all (alpha=2.333) ===")
    fasttext = load_text_embeddings(
        PROJECT_ROOT / "models/cc.fr.300.vec", max_words=300000
    )
    graph_fr = build_wolf_graph(
        PROJECT_ROOT / "datasets/wolf/wolf-1.0b4.xml",
        include_synonyms=True, include_hypernyms=True, include_hyponyms=True,
        keep_only_vocab=False
    )
    graph_fr = filter_graph_by_vocab(graph_fr, set(fasttext))
    retrofitted_fr, _ = retrofit_vectors(fasttext, graph_fr, num_iters=10, alpha=2.333)

    make_viz(
        fasttext, retrofitted_fr, FR_WORD_GROUPS, GROUP_COLORS_FR,
        "Word Vector Movement After Retrofitting (PCA 2D)\n"
        "French: fastText + WOLF (syn+hyper+hypo), alpha=2.333",
        "results/eval_viz_french_best.png"
    )

    print("\n=== Both visualizations done ===")


if __name__ == "__main__":
    main()