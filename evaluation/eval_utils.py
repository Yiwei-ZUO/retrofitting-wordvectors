import numpy as np
from scipy.stats import spearmanr


def load_ws353(path):
    """Load WS-353: word1,word2,score (with header)"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from WS-353")
    return pairs


def load_simlex(path):
    """Load SimLex-999: tab separated, column 4 is score (with header)"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[3])))
    print(f"Loaded {len(pairs)} pairs from SimLex-999")
    return pairs


def load_rg65(path):
    """Load RG-65: tab separated, no header"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from RG-65")
    return pairs


def load_fr_simlex(path):
    """Load French SimLex: semicolon separated, with header."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 4:
                continue
            pairs.append((parts[1].lower(), parts[2].lower(), float(parts[3])))
    print(f"Loaded {len(pairs)} pairs from French SimLex")
    return pairs


def load_fr_ws353(path):
    """Load French WS-353: semicolon separated, with header."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 4:
                continue
            pairs.append((parts[1].lower(), parts[2].lower(), float(parts[3])))
    print(f"Loaded {len(pairs)} pairs from French WS-353")
    return pairs


def load_rg65_french(path):
    """Load RG65_FR: space separated, no header."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
    print(f"Loaded {len(pairs)} pairs from RG65_FR")
    return pairs

def cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def evaluate_with_coverage(vectors: dict, pairs: list) -> dict:
    """Evaluate word similarity, return rho plus coverage info."""
    human_scores, model_scores = [], []
    skipped = 0
    for w1, w2, human_score in pairs:
        if w1 not in vectors or w2 not in vectors:
            skipped += 1
            continue
        human_scores.append(human_score)
        model_scores.append(cosine_similarity(vectors[w1], vectors[w2]))
    covered = len(human_scores)
    total = len(pairs)
    rho = float("nan") if covered < 2 else float(spearmanr(human_scores, model_scores)[0])
    return {"rho": rho, "covered": covered, "total": total,
            "skipped": skipped, "coverage": covered / total if total else 0.0}
