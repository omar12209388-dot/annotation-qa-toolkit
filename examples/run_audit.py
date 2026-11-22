#!/usr/bin/env python3
"""Run the demo second-level QA audit over the synthetic batch.

Usage:
    python data/generate_synthetic.py   # once, to build the CSVs
    python examples/run_audit.py

Writes examples/output/audit_report.json and audit_report.md.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_qa.agreement import krippendorff_alpha, ratings_to_fleiss_matrix, fleiss_kappa
from annotation_qa.audit import audit_annotators, load_annotations, load_gold
from annotation_qa.pii import find_pii
from annotation_qa.sampling import risk_based_sample

ACCURACY_FLOOR = 0.85


def main() -> int:
    annotations = load_annotations(ROOT / "data" / "annotations.csv")
    gold = load_gold(ROOT / "data" / "gold.csv")

    # --- Agreement across the two raters per item (Krippendorff + Fleiss)
    by_item: dict[str, dict[str, str]] = defaultdict(dict)
    for ann in annotations:
        by_item[ann.item_id][ann.annotator] = ann.label
    items = sorted(by_item)
    raters = sorted({a.annotator for a in annotations})
    rater_columns = [
        [by_item[i].get(r) for i in items] for r in raters
    ]
    alpha = krippendorff_alpha(*rater_columns)
    matrix = ratings_to_fleiss_matrix([list(by_item[i].values()) for i in items])

    # --- Gold-set accuracy audit
    audits = audit_annotators(annotations, gold, accuracy_floor=ACCURACY_FLOOR)
    flagged = [a.annotator for a in audits if a.flagged]

    # --- PII screening of the annotated texts
    pii_hits = []
    seen = set()
    for ann in annotations:
        if ann.item_id in seen:
            continue
        seen.add(ann.item_id)
        for match in find_pii(ann.text):
            pii_hits.append({"item_id": ann.item_id, "kind": match.kind,
                             "value": match.value})

    # --- Recommended audit sample for the next QA pass
    sample = risk_based_sample(annotations, flagged, target=15)

    overall_accuracy = sum(a.correct for a in audits) / sum(a.gold_items for a in audits)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "items": len(by_item),
            "annotations": len(annotations),
            "annotators": len(raters),
            "gold_labels": len(gold),
            "source": "synthetic demo batch (data/generate_synthetic.py)",
        },
        "agreement": {
            "krippendorff_alpha_nominal": round(alpha, 3),
            "fleiss_kappa": round(fleiss_kappa(matrix), 3),
        },
        "overall_gold_accuracy": round(overall_accuracy, 3),
        "annotator_audits": [
            {
                "annotator": a.annotator,
                "gold_items": a.gold_items,
                "accuracy": a.accuracy,
                "confusions": a.confusions,
                "flagged": a.flagged,
                "flag_reasons": a.flag_reasons,
            }
            for a in audits
        ],
        "pii_hits": pii_hits,
        "recommended_audit_sample": sample,
    }

    out = ROOT / "examples" / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Annotation QA Audit Report",
        "",
        f"Generated {report['generated_at']} · synthetic demonstration data",
        "",
        f"- Items: {len(by_item)} (each labeled by 2 annotators), gold labels: {len(gold)}",
        f"- Krippendorff's alpha (nominal): {alpha:.3f}",
        f"- Fleiss' kappa: {fleiss_kappa(matrix):.3f}",
        f"- Overall gold accuracy: {overall_accuracy:.1%}",
        "",
        "## Annotator audits",
        "",
        "| Annotator | Gold items | Accuracy | Flagged | Notes |",
        "|---|---|---|---|---|",
    ]
    for a in audits:
        notes = "; ".join(a.flag_reasons) if a.flag_reasons else "ok"
        top = ", ".join(list(a.confusions)[:2])
        if top:
            notes += f" (top confusions: {top})"
        lines.append(
            f"| {a.annotator} | {a.gold_items} | {a.accuracy:.1%} | "
            f"{'YES' if a.flagged else 'no'} | {notes} |"
        )
    lines += [
        "",
        "## PII screening",
        "",
    ]
    if pii_hits:
        lines += ["| Item | Type | Value |", "|---|---|---|"]
        lines += [f"| {h['item_id']} | {h['kind']} | `{h['value']}` |" for h in pii_hits]
    else:
        lines.append("No PII patterns matched.")
    lines += [
        "",
        f"## Recommended next audit sample ({len(sample)} items)",
        "",
        "Disagreement items and flagged-annotator items first, then random fill:",
        "",
        "`" + "`, `".join(sample) + "`",
        "",
    ]
    (out / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"alpha={alpha:.3f}  fleiss={fleiss_kappa(matrix):.3f}  "
          f"overall accuracy={overall_accuracy:.1%}")
    for a in audits:
        mark = "FLAGGED" if a.flagged else "ok"
        print(f"  {a.annotator:12s} accuracy {a.accuracy:.1%}  {mark}")
    print(f"pii hits: {len(pii_hits)}  audit sample: {len(sample)} items")
    print(f"reports written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
