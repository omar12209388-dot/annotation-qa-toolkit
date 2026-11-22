"""Inter-annotator agreement statistics.

Cohen's kappa (two raters), Fleiss' kappa (n raters) and Krippendorff's
alpha (nominal), implemented from first principles with no dependencies.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Sequence


def cohen_kappa(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must have equal length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("no items")
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    cats = set(labels_a) | set(labels_b)
    ca, cb = Counter(labels_a), Counter(labels_b)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return 1.0 if p_e == 1.0 else (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(matrix: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa from an items x categories count matrix.

    Each row holds, for one item, how many raters assigned each category.
    """
    n_items = len(matrix)
    if n_items == 0:
        raise ValueError("no items")
    n_raters = sum(matrix[0])
    if n_raters < 2:
        raise ValueError("need at least two raters")
    k = len(matrix[0])

    p_items = []
    for row in matrix:
        if sum(row) != n_raters:
            raise ValueError("all rows must sum to the rater count")
        p_items.append(
            (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        )
    p_bar = sum(p_items) / n_items

    p_cat = [sum(row[j] for row in matrix) / (n_items * n_raters) for j in range(k)]
    p_e = sum(p * p for p in p_cat)
    return 1.0 if p_e == 1.0 else (p_bar - p_e) / (1.0 - p_e)


def krippendorff_alpha(*rater_labels: Sequence[Hashable]) -> float:
    """Krippendorff's alpha (nominal) for any number of raters.

    Standard coincidence-matrix construction: for each unit with m
    pairable values, every ordered value pair contributes 1/(m - 1).
    Raters may have missing values (None); units with fewer than two
    values are skipped. Matches the reference `krippendorff` package.
    """
    if len(rater_labels) < 2:
        raise ValueError("need at least two raters")
    n_items = len(rater_labels[0])
    if any(len(r) != n_items for r in rater_labels):
        raise ValueError("label sequences must have equal length")

    coincidence: Counter = Counter()  # (c, k) -> weighted count, ordered pairs
    for i in range(n_items):
        vals = [r[i] for r in rater_labels if r[i] is not None]
        m = len(vals)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    coincidence[(vals[a], vals[b])] += 1.0 / (m - 1)

    n_total = sum(coincidence.values()) / 2  # ordered pairs counted twice
    if n_total == 0:
        raise ValueError("no ratable items")

    # Marginal totals per value.
    n_c: Counter = Counter()
    for (c, k), v in coincidence.items():
        n_c[c] += v

    # Observed disagreement (nominal metric: delta = 1 when c != k).
    d_o = sum(v for (c, k), v in coincidence.items() if c != k) / (2 * n_total)
    # Expected disagreement from the marginals.
    d_e = 0.0
    vals = list(n_c)
    for c in vals:
        for k in vals:
            if c != k:
                d_e += n_c[c] * n_c[k]
    d_e /= 2 * n_total * (2 * n_total - 1)
    return 1.0 if d_e == 0.0 else 1.0 - d_o / d_e


def ratings_to_fleiss_matrix(
    item_ratings: Sequence[Sequence[Hashable]],
) -> list[list[int]]:
    """Build a Fleiss matrix from per-item rating lists.

    Args:
        item_ratings: one inner list per item, with that item's ratings.
    """
    cats = sorted({r for ratings in item_ratings for r in ratings}, key=str)
    idx = {c: i for i, c in enumerate(cats)}
    matrix = []
    for ratings in item_ratings:
        row = [0] * len(cats)
        for r in ratings:
            row[idx[r]] += 1
        matrix.append(row)
    return matrix
