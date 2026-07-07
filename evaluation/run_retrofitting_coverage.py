from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_utils import (
    load_fr_simlex,
    load_fr_ws353,
    load_rg65,
    load_rg65_french,
    load_simlex,
    load_ws353,
)
from src.preprocessing import (
    build_ppdb_graph,
    build_wolf_graph,
    build_wordnet_graph,
    filter_graph_by_vocab,
)
from src.utils import load_text_embeddings, load_word2vec_binary


GLOVE_PATH = PROJECT_ROOT / "models/glove.6B.300d.txt"
WORD2VEC_PATH = PROJECT_ROOT / "models/GoogleNews-vectors-negative300.bin.gz"
FASTTEXT_PATH = PROJECT_ROOT / "models/cc.fr.300.vec"
PPDB_PATH = PROJECT_ROOT / "datasets/ppdb/ppdb-xl.txt"
WOLF_PATH = PROJECT_ROOT / "datasets/wolf/wolf-1.0b4.xml"


def count_coverage(combo_name, dataset_name, pairs, embedding_vocab, graph_vocab):
    unique_words = set()
    embedding_words = set()
    retrofitted_words = set()

    embedding_covered_pairs = 0
    retrofitted_pairs = 0
    at_least_one_retrofitted_pairs = 0

    for word1, word2, _score in pairs:
        unique_words.update((word1, word2))

        word1_has_embedding = word1 in embedding_vocab
        word2_has_embedding = word2 in embedding_vocab
        word1_retrofitted = word1 in graph_vocab
        word2_retrofitted = word2 in graph_vocab

        if word1_has_embedding:
            embedding_words.add(word1)
        if word2_has_embedding:
            embedding_words.add(word2)

        if word1_retrofitted:
            retrofitted_words.add(word1)
        if word2_retrofitted:
            retrofitted_words.add(word2)

        if word1_has_embedding and word2_has_embedding:
            embedding_covered_pairs += 1

        if word1_retrofitted and word2_retrofitted:
            retrofitted_pairs += 1

        if word1_retrofitted or word2_retrofitted:
            at_least_one_retrofitted_pairs += 1

    total_pairs = len(pairs)
    unique_count = len(unique_words)

    return {
        "combo": combo_name,
        "dataset": dataset_name,
        "total_pairs": total_pairs,
        "embedding_covered_pairs": embedding_covered_pairs,
        "embedding_pair_coverage": round(
            embedding_covered_pairs / total_pairs, 4
        ) if total_pairs else 0.0,
        "both_words_retrofitted_pairs": retrofitted_pairs,
        "both_words_retrofitted_pair_coverage": round(
            retrofitted_pairs / total_pairs, 4
        ) if total_pairs else 0.0,
        "at_least_one_word_retrofitted_pairs": at_least_one_retrofitted_pairs,
        "unique_words": unique_count,
        "embedding_covered_unique_words": len(embedding_words),
        "actually_retrofitted_unique_words": len(retrofitted_words),
        "actually_retrofitted_unique_word_coverage": round(
            len(retrofitted_words) / unique_count, 4
        ) if unique_count else 0.0,
    }


def load_english_datasets():
    return {
        "SimLex-999": load_simlex(PROJECT_ROOT / "datasets/en-simlex.txt"),
        "WS-353": load_ws353(PROJECT_ROOT / "datasets/en-ws353.csv"),
        "RG-65": load_rg65(PROJECT_ROOT / "datasets/en_rg65.txt"),
    }


def load_french_datasets():
    return {
        "SimLex-999_FR": load_fr_simlex(PROJECT_ROOT / "datasets/fr-simlex.dataset"),
        "WS-353_FR": load_fr_ws353(PROJECT_ROOT / "datasets/fr-ws353.dataset"),
        "RG-65_FR": load_rg65_french(PROJECT_ROOT / "datasets/fr_rg65.txt"),
    }


def load_vocab_from_text_embeddings(path, max_words):
    embeddings = load_text_embeddings(path, max_words=max_words)
    return set(embeddings)


def load_vocab_from_word2vec(path, max_words):
    embeddings = load_word2vec_binary(path, max_words=max_words)
    return set(embeddings)


def add_rows(rows, combo_name, datasets, embedding_vocab, graph):
    graph_vocab = set(graph)
    for dataset_name, pairs in datasets.items():
        rows.append(
            count_coverage(
                combo_name,
                dataset_name,
                pairs,
                embedding_vocab,
                graph_vocab,
            )
        )


def run_glove(rows, datasets, max_words):
    if not GLOVE_PATH.exists():
        print(f"Skipping GloVe: missing {GLOVE_PATH}")
        return

    print("\nLoading GloVe vocabulary...")
    vocab = load_vocab_from_text_embeddings(GLOVE_PATH, max_words)
    print(f"GloVe vocabulary size: {len(vocab)}")

    print("Building GloVe + WordNet synonym graph...")
    graph = build_wordnet_graph(
        vocab,
        include_synonyms=True,
        include_hypernyms=False,
        include_hyponyms=False,
        keep_only_vocab=False,
    )
    graph = filter_graph_by_vocab(graph, vocab)
    add_rows(rows, "GloVe+WN_syn", datasets, vocab, graph)

    print("Building GloVe + WordNet all-relations graph...")
    graph = build_wordnet_graph(
        vocab,
        include_synonyms=True,
        include_hypernyms=True,
        include_hyponyms=True,
        keep_only_vocab=False,
    )
    graph = filter_graph_by_vocab(graph, vocab)
    add_rows(rows, "GloVe+WN_all", datasets, vocab, graph)

    if PPDB_PATH.exists():
        print("Building GloVe + PPDB graph...")
        graph = build_ppdb_graph(PPDB_PATH, keep_only_vocab=False)
        graph = filter_graph_by_vocab(graph, vocab)
        add_rows(rows, "GloVe+PPDB", datasets, vocab, graph)
    else:
        print(f"Skipping PPDB: missing {PPDB_PATH}")


def run_word2vec(rows, datasets, max_words):
    if not WORD2VEC_PATH.exists():
        print(f"Skipping Word2Vec: missing {WORD2VEC_PATH}")
        return

    print("\nLoading Word2Vec vocabulary...")
    vocab = load_vocab_from_word2vec(WORD2VEC_PATH, max_words)
    print(f"Word2Vec vocabulary size: {len(vocab)}")

    print("Building Word2Vec + WordNet synonym graph...")
    graph = build_wordnet_graph(
        vocab,
        include_synonyms=True,
        include_hypernyms=False,
        include_hyponyms=False,
        keep_only_vocab=False,
    )
    graph = filter_graph_by_vocab(graph, vocab)
    add_rows(rows, "Word2Vec+WN_syn", datasets, vocab, graph)

    print("Building Word2Vec + WordNet all-relations graph...")
    graph = build_wordnet_graph(
        vocab,
        include_synonyms=True,
        include_hypernyms=True,
        include_hyponyms=True,
        keep_only_vocab=False,
    )
    graph = filter_graph_by_vocab(graph, vocab)
    add_rows(rows, "Word2Vec+WN_all", datasets, vocab, graph)


def run_french(rows, datasets, max_words):
    if not FASTTEXT_PATH.exists():
        print(f"Skipping French fastText: missing {FASTTEXT_PATH}")
        return
    if not WOLF_PATH.exists():
        print(f"Skipping WOLF: missing {WOLF_PATH}")
        return

    print("\nLoading French fastText vocabulary...")
    vocab = load_vocab_from_text_embeddings(FASTTEXT_PATH, max_words)
    print(f"French fastText vocabulary size: {len(vocab)}")

    print("Building fastText + WOLF synonym graph...")
    graph = build_wolf_graph(
        WOLF_PATH,
        include_synonyms=True,
        include_hypernyms=False,
        include_hyponyms=False,
        keep_only_vocab=False,
    )
    graph = filter_graph_by_vocab(graph, vocab)
    add_rows(rows, "fastText+WOLF_syn", datasets, vocab, graph)

    print("Building fastText + WOLF all-relations graph...")
    graph = build_wolf_graph(
        WOLF_PATH,
        include_synonyms=True,
        include_hypernyms=True,
        include_hyponyms=True,
        keep_only_vocab=False,
    )
    graph = filter_graph_by_vocab(graph, vocab)
    add_rows(rows, "fastText+WOLF_all", datasets, vocab, graph)


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "combo",
        "dataset",
        "total_pairs",
        "embedding_covered_pairs",
        "embedding_pair_coverage",
        "both_words_retrofitted_pairs",
        "both_words_retrofitted_pair_coverage",
        "at_least_one_word_retrofitted_pairs",
        "unique_words",
        "embedding_covered_unique_words",
        "actually_retrofitted_unique_words",
        "actually_retrofitted_unique_word_coverage",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Count how many evaluation pairs are actually affected by retrofitting."
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional vocabulary limit for quick tests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. If omitted, results are printed only.",
    )
    args = parser.parse_args()

    rows = []
    english_datasets = load_english_datasets()
    french_datasets = load_french_datasets()

    run_glove(rows, english_datasets, args.max_words)
    run_word2vec(rows, english_datasets, args.max_words)
    run_french(rows, french_datasets, args.max_words)

    print("\nCoverage summary:")
    for row in rows:
        print(
            f"{row['combo']} | {row['dataset']} | "
            f"embedding pairs: {row['embedding_covered_pairs']}/{row['total_pairs']} | "
            f"both words retrofitted: "
            f"{row['both_words_retrofitted_pairs']}/{row['total_pairs']} | "
            f"retrofitted unique words: "
            f"{row['actually_retrofitted_unique_words']}/{row['unique_words']}"
        )

    if args.output is not None:
        write_csv(rows, args.output)
        print(f"\nCoverage CSV saved to {args.output}")


if __name__ == "__main__":
    main()
