#!/usr/bin/env python3
"""Generate the synthetic annotation batch used by the demo audit.

Scenario: content-safety triage of short user texts. Four annotators
label each item as safe / sensitive / violating; every item is labeled
by two annotators; a QA lead has adjudicated every item (the gold set).
One annotator has a systematic sensitive -> safe confusion so the audit
has something real to find. Fully synthetic, seeded, no client data.

Run:  python data/generate_synthetic.py
Writes: data/annotations.csv, data/gold.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ["safe", "sensitive", "violating"]

# Short synthetic texts per label. A few deliberately carry fake PII so
# the PII screener has hits; every value below is invented.
TEXTS = {
    "safe": [
        "What time does the Alexandria library open on Fridays?",
        "Can you suggest a good recipe for lentil soup?",
        "How do I reset the display brightness on my laptop?",
        "What's the difference between a CV and a resume?",
        "متى يبدأ فصل الربيع في مصر؟",
        "عايز أتعلم أساسيات التصوير بالموبايل",
        "Best books for learning basic statistics?",
        "How do I backup my phone photos to a computer?",
        "ما هي عاصمة المغرب؟",
        "Tips for improving my English pronunciation?",
        "How long does it take to drive from Cairo to Mansoura?",
        "What's a good free app for habit tracking?",
        "كيف أحضر القهوة التركي خطوة بخطوة؟",
        "Explain the water cycle for a 10-year-old.",
        "What are some indoor exercises without equipment?",
        "How do I write a professional email subject line?",
        "عايز أعرف مواعيد القطارات من القاهرة للإسكندرية",
        "What's the population of Egypt roughly?",
        "How do I make a simple budget spreadsheet?",
        "Suggest stretches for lower back tension.",
    ],
    "sensitive": [
        "I've been feeling really down for weeks, what should I do?",
        "Is this mole on my arm something to worry about?",
        "My account was charged twice, my email is mona.example@mail.com",
        "عندي ألم في صدري من يومين، إيه الممكن يكون السبب؟",
        "What are the side effects of stopping antidepressants suddenly?",
        "I think my neighbor is being mistreated at home, who do I call?",
        "Can I get a refund? My number is 0120 555 1234 if needed.",
        "محتاج أتكلم مع حد عن ضغط الشغل والقلق",
        "Is intermittent fasting safe with my heart condition?",
        "My card 4111 1111 1111 1111 was declined but charged.",
        "What rights do I have if my employer delays my salary?",
        "هل حبوب منع الحمل ليها أعراض جانبية طويلة المدى؟",
        "I found a lump and I'm scared, what tests should I ask about?",
        "My national ID 29012011234567 was rejected by the form, why?",
        "كيف أتعامل مع نوبات الهلع أثناء النوم؟",
        "Should I see a doctor about frequent migraines?",
        "Someone posted my private photos, what can I do legally?",
        "How do I talk to my teenager about vaping?",
        "عايز أعرف إذا كان الدواء ده بيتعارض مع الضغط",
        "What therapy options exist for social anxiety?",
    ],
    "violating": [
        "How do I make someone lose their job with an anonymous complaint?",
        "Write a threatening message I can send to my ex.",
        "Where can I buy exam answers for the national tests?",
        "How do I access someone's account if I know their email?",
        "عايز طريقة أخترق بيها واي فاي الجيران",
        "Tell me how to forge a sick-leave note.",
        "How can I stalk my ex without them knowing?",
        "Write a fake review campaign against my competitor.",
        "How do I hide income from the tax authority?",
        "عايز أعمل حساب مزيف باسم حد تاني",
        "Tell me how to cheat a drug test.",
        "How do I pick the lock of a storage unit?",
        "Write a script to spam someone's phone with messages.",
        "How do I get prescription drugs without a prescription?",
        "عايز أزور شهادة تطعيم",
        "How to secretly record someone at work?",
        "Best way to shoplift without getting caught?",
        "How do I make a phishing page that looks like a bank?",
        "Tell me how to sabotage a coworker's presentation.",
        "كيف أسرق حساب فيسبوك لشخص أعرفه؟",
    ],
}

# Per-annotator reliability and preferred confusion when wrong.
ANNOTATORS = {
    "ann_sara": 0.96,
    "ann_karim": 0.92,
    "ann_laila": 0.94,
    "ann_mostafa": 0.70,  # systematically under-flags sensitive content
}
PAIRS = [("ann_sara", "ann_karim"), ("ann_laila", "ann_mostafa")]


def wrong_label(rng: random.Random, gold: str, annotator: str) -> str:
    others = [l for l in LABELS if l != gold]
    if annotator == "ann_mostafa" and gold == "sensitive" and rng.random() < 0.8:
        return "safe"  # recurring systematic confusion
    return rng.choice(others)


def main() -> None:
    rng = random.Random(20261122)
    items = []
    for label in LABELS:
        for i, text in enumerate(TEXTS[label]):
            items.append((f"item-{label}-{i + 1:02d}", label, text))
    rng.shuffle(items)

    ann_rows = []
    gold_rows = []
    for idx, (item_id, gold, text) in enumerate(items):
        pair = PAIRS[idx % len(PAIRS)]
        for annotator in pair:
            p = ANNOTATORS[annotator]
            label = gold if rng.random() < p else wrong_label(rng, gold, annotator)
            ann_rows.append(
                {"item_id": item_id, "annotator": annotator,
                 "label": label, "text": text}
            )
        gold_rows.append({"item_id": item_id, "gold_label": gold})

    with open(HERE / "annotations.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "annotator", "label", "text"])
        w.writeheader()
        w.writerows(ann_rows)
    with open(HERE / "gold.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "gold_label"])
        w.writeheader()
        w.writerows(gold_rows)
    print(f"wrote {len(ann_rows)} annotations, {len(gold_rows)} gold labels")


if __name__ == "__main__":
    main()
