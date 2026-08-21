"""
Evaluation metrics for 3D spatial question answering (ScanQA-style).

Implements, from scratch and with no heavy external NLP dependencies:
  - Exact Match (EM)
  - BLEU-4 (with brevity penalty)
  - CIDEr (TF-IDF weighted n-gram cosine similarity, corpus-level)
  - Question-type sub-metrics: Spatial-Relation Accuracy, Counting Accuracy,
    Existence Accuracy

This corresponds to Section 3.8.1 of the thesis (Table 3.2 and the three
sub-metrics described immediately after it).
"""

import math
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------------
# Text normalisation / tokenisation
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lower-case and strip punctuation, matching the EM definition in 3.8.1."""
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def compute_em(prediction: str, references: Sequence[str]) -> float:
    """1.0 if the normalised prediction matches ANY normalised reference, else 0.0."""
    pred_norm = normalize_answer(prediction)
    return 1.0 if any(pred_norm == normalize_answer(ref) for ref in references) else 0.0


# ---------------------------------------------------------------------------
# BLEU-4
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu4(prediction: str, references: Sequence[str]) -> float:
    """Standard BLEU-4: modified n-gram precision (n=1..4) with brevity penalty.

    Uses the reference with the closest length, as is standard practice, and
    takes the max modified precision against all references for each n-gram
    order (multi-reference BLEU).
    """
    pred_tokens = tokenize(prediction)
    ref_token_lists = [tokenize(r) for r in references]

    if len(pred_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        pred_ngrams = _ngrams(pred_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue

        max_ref_counts = Counter()
        for ref_tokens in ref_token_lists:
            ref_ngrams = _ngrams(ref_tokens, n)
            for gram, count in ref_ngrams.items():
                max_ref_counts[gram] = max(max_ref_counts[gram], count)

        clipped = sum(min(count, max_ref_counts[gram]) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if min(precisions) == 0.0:
        # standard smoothing: if any order has zero precision, BLEU-4 is 0
        # (a light smoothing epsilon is used instead of a hard zero, which is
        # friendlier for short ScanQA-style answers)
        precisions = [p if p > 0 else 1e-9 for p in precisions]

    log_avg = sum(math.log(p) for p in precisions) / 4.0
    geo_mean = math.exp(log_avg)

    pred_len = len(pred_tokens)
    closest_ref_len = min(
        (len(rt) for rt in ref_token_lists),
        key=lambda ref_len: (abs(ref_len - pred_len), ref_len),
        default=pred_len,
    )
    if pred_len > closest_ref_len:
        brevity_penalty = 1.0
    elif pred_len == 0:
        brevity_penalty = 0.0
    else:
        brevity_penalty = math.exp(1 - closest_ref_len / pred_len)

    return brevity_penalty * geo_mean


# ---------------------------------------------------------------------------
# CIDEr
# ---------------------------------------------------------------------------

def _cider_ngram_counts(tokens: List[str], n: int) -> Counter:
    return _ngrams(tokens, n)


def compute_cider(
    predictions: Sequence[str],
    references_list: Sequence[Sequence[str]],
    n_max: int = 4,
) -> List[float]:
    """Corpus-level CIDEr. Returns one score per (prediction, references) pair.

    CIDEr needs the whole corpus to compute document frequencies (for TF-IDF),
    so it is a batch function rather than a per-example one like EM/BLEU-4.
    """
    num_docs = len(predictions)
    assert num_docs == len(references_list)

    # Document frequency: for each n, how many reference groups contain this n-gram.
    doc_freq = [defaultdict(int) for _ in range(n_max)]
    for references in references_list:
        seen_per_n = [set() for _ in range(n_max)]
        for ref in references:
            ref_tokens = tokenize(ref)
            for n in range(1, n_max + 1):
                for gram in _cider_ngram_counts(ref_tokens, n):
                    seen_per_n[n - 1].add(gram)
        for n in range(n_max):
            for gram in seen_per_n[n]:
                doc_freq[n][gram] += 1

    def tfidf_vector(tokens: List[str], n: int) -> Dict[tuple, float]:
        counts = _cider_ngram_counts(tokens, n)
        total = sum(counts.values())
        vec = {}
        for gram, count in counts.items():
            tf = count / total if total > 0 else 0.0
            df = doc_freq[n - 1].get(gram, 0)
            idf = math.log(max(num_docs, 1) / max(df, 1))
            vec[gram] = tf * idf
        return vec

    def cosine_sim(vec_a: Dict[tuple, float], vec_b: Dict[tuple, float]) -> float:
        common = set(vec_a) & set(vec_b)
        numerator = sum(vec_a[g] * vec_b[g] for g in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return numerator / (norm_a * norm_b)

    scores = []
    for prediction, references in zip(predictions, references_list):
        pred_tokens = tokenize(prediction)
        per_n_scores = []
        for n in range(1, n_max + 1):
            pred_vec = tfidf_vector(pred_tokens, n)
            ref_sims = []
            for ref in references:
                ref_vec = tfidf_vector(tokenize(ref), n)
                ref_sims.append(cosine_sim(pred_vec, ref_vec))
            per_n_scores.append(sum(ref_sims) / len(ref_sims) if ref_sims else 0.0)
        # average over n=1..4, scaled by 10 to match the conventional CIDEr range
        scores.append(10.0 * sum(per_n_scores) / n_max)

    return scores


# ---------------------------------------------------------------------------
# Question-type classification for the Section 3.8.1 sub-metrics
# ---------------------------------------------------------------------------

_SPATIAL_RELATION_TERMS = [
    "next to", "in front of", "behind", "above", "below", "between",
    "left of", "right of", "on the left", "on the right", "near",
    "beside", "under", "over", "adjacent to", "across from",
]

_COUNTING_PATTERN = re.compile(r"\bhow many\b", re.IGNORECASE)
_EXISTENCE_PATTERN = re.compile(r"^\s*is there\b", re.IGNORECASE)


def classify_question(question: str) -> str:
    """Classify a ScanQA question into one of the sub-metric categories.

    Returns one of: 'spatial_relation', 'counting', 'existence', 'other'.
    Checked in this order because a question could in principle contain both
    a counting phrase and a spatial term; counting and existence patterns are
    the most syntactically distinctive, so they are checked first.
    """
    q_lower = question.lower()
    if _COUNTING_PATTERN.search(q_lower):
        return "counting"
    if _EXISTENCE_PATTERN.search(q_lower):
        return "existence"
    if any(term in q_lower for term in _SPATIAL_RELATION_TERMS):
        return "spatial_relation"
    return "other"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: Sequence[str],
    references_list: Sequence[Sequence[str]],
    questions: Sequence[str],
) -> Dict[str, float]:
    """Compute the full Section 3.8.1 metric suite over a batch of predictions.

    Args:
        predictions: model-generated answer strings, one per example.
        references_list: for each example, one or more reference (gold) answers.
        questions: the original question text for each example, used to
            compute the question-type-conditioned sub-metrics.

    Returns:
        A flat dict with overall EM / BLEU-4 / CIDEr plus the three sub-metrics
        and their support (number of questions each sub-metric was computed
        over), matching the reporting used in Table 3.2 / Table 3.4.
    """
    assert len(predictions) == len(references_list) == len(questions)
    n = len(predictions)

    em_scores = [compute_em(p, r) for p, r in zip(predictions, references_list)]
    bleu_scores = [compute_bleu4(p, r) for p, r in zip(predictions, references_list)]
    cider_scores = compute_cider(predictions, references_list)

    categories = [classify_question(q) for q in questions]

    results: Dict[str, float] = {
        "EM": sum(em_scores) / n if n else 0.0,
        "BLEU-4": sum(bleu_scores) / n if n else 0.0,
        "CIDEr": sum(cider_scores) / n if n else 0.0,
        "num_examples": n,
    }

    for category, metric_name in [
        ("spatial_relation", "Spatial-Relation Accuracy"),
        ("counting", "Counting Accuracy"),
        ("existence", "Existence Accuracy"),
    ]:
        indices = [i for i, c in enumerate(categories) if c == category]
        if indices:
            results[metric_name] = sum(em_scores[i] for i in indices) / len(indices)
        else:
            results[metric_name] = float("nan")
        results[f"{metric_name} (n)"] = len(indices)

    return results