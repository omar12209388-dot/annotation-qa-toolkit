"""Audit sampling: which items a QA lead should re-review.

Strategies:
- random: uniform sample, seeded for reproducibility.
- stratified: a fixed share per annotator, so no one escapes audit.
- risk_based: every item whose annotators disagreed, plus items from
  flagged annotators, then random fill to the target size.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Sequence


def random_sample(item_ids: Sequence[str], n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    ids = list(item_ids)
    rng.shuffle(ids)
    return sorted(ids[: max(0, min(n, len(ids)))])


def stratified_sample(
    annotations: Sequence, share: float, seed: int = 42
) -> list[str]:
    """Sample the same share of each annotator's items."""
    rng = random.Random(seed)
    by_annotator: dict[str, list[str]] = defaultdict(list)
    for ann in annotations:
        by_annotator[ann.annotator].append(ann.item_id)
    picked: list[str] = []
    for items in by_annotator.values():
        items = list(items)
        rng.shuffle(items)
        k = max(1, round(len(items) * share))
        picked.extend(items[:k])
    return sorted(set(picked))


def risk_based_sample(
    annotations: Sequence,
    flagged_annotators: Sequence[str],
    target: int,
    seed: int = 42,
) -> list[str]:
    """Prioritize disagreement items and flagged annotators' work."""
    by_item: dict[str, list] = defaultdict(list)
    for ann in annotations:
        by_item[ann.item_id].append(ann)

    disagreement: list[str] = []
    flagged_only: list[str] = []
    for item_id, anns in by_item.items():
        labels = {a.label for a in anns}
        if len(labels) > 1:  # annotators disagreed on this item
            disagreement.append(item_id)
        elif any(a.annotator in flagged_annotators for a in anns):
            flagged_only.append(item_id)

    # Priority order: disagreements, then other flagged-annotator work,
    # then a random fill, capped at target.
    ranked = sorted(disagreement) + sorted(flagged_only)
    chosen = list(ranked[:target])
    if len(chosen) < target:
        rest = [i for i in by_item if i not in set(chosen)]
        rng = random.Random(seed)
        rng.shuffle(rest)
        chosen.extend(rest[: target - len(chosen)])
    return sorted(set(chosen))
