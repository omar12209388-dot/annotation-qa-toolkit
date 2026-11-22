"""Gold-set accuracy audits: per-annotator accuracy, confusion, and flags.

This is the second-level QA view: given annotations plus a gold subset
that was reviewed by a QA lead, measure each annotator, find the label
pairs they confuse, and flag the ones who need a guideline refresh.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Annotation:
    item_id: str
    annotator: str
    label: str
    text: str = ""


@dataclass
class AnnotatorAudit:
    annotator: str
    gold_items: int
    correct: int
    accuracy: float
    confusions: dict[str, int] = field(default_factory=dict)  # "gold->label": n
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)


def load_annotations(path: str | Path) -> list[Annotation]:
    """Load a CSV with columns item_id, annotator, label[, text]."""
    rows: list[Annotation] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                Annotation(
                    item_id=row["item_id"],
                    annotator=row["annotator"],
                    label=row["label"],
                    text=row.get("text", ""),
                )
            )
    return rows


def load_gold(path: str | Path) -> dict[str, str]:
    """Load a CSV with columns item_id, gold_label."""
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["item_id"]: row["gold_label"] for row in csv.DictReader(fh)}


def audit_annotators(
    annotations: list[Annotation],
    gold: dict[str, str],
    min_gold_items: int = 5,
    accuracy_floor: float = 0.85,
) -> list[AnnotatorAudit]:
    """Score every annotator against the gold subset.

    An annotator is flagged when their gold accuracy falls below
    `accuracy_floor` (with at least `min_gold_items` gold items) or when
    a single confusion pair accounts for at least half of their errors.
    """
    per_annotator: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        if ann.item_id in gold:
            per_annotator[ann.annotator].append(ann)

    audits: list[AnnotatorAudit] = []
    for name, anns in sorted(per_annotator.items()):
        correct = 0
        confusion: Counter = Counter()
        for ann in anns:
            gold_label = gold[ann.item_id]
            if ann.label == gold_label:
                correct += 1
            else:
                confusion[f"{gold_label} -> {ann.label}"] += 1
        n = len(anns)
        accuracy = correct / n
        audit = AnnotatorAudit(
            annotator=name,
            gold_items=n,
            correct=correct,
            accuracy=round(accuracy, 3),
            confusions=dict(confusion.most_common()),
        )
        errors = n - correct
        if n >= min_gold_items and accuracy < accuracy_floor:
            audit.flagged = True
            audit.flag_reasons.append(
                f"accuracy {accuracy:.1%} below floor {accuracy_floor:.0%}"
            )
        if errors:
            top_pair, top_n = confusion.most_common(1)[0]
            if top_n >= 3 and top_n >= errors / 2:
                audit.flagged = True
                audit.flag_reasons.append(
                    f"recurring confusion {top_pair} ({top_n} of {errors} errors)"
                )
        audits.append(audit)
    return audits
