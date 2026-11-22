# Annotation QA Audit Report

Generated 2026-11-22T18:35:04+00:00 · synthetic demonstration data

- Items: 60 (each labeled by 2 annotators), gold labels: 60
- Krippendorff's alpha (nominal): 0.752
- Fleiss' kappa: 0.750
- Overall gold accuracy: 91.7%

## Annotator audits

| Annotator | Gold items | Accuracy | Flagged | Notes |
|---|---|---|---|---|
| ann_karim | 30 | 96.7% | no | ok (top confusions: violating -> sensitive) |
| ann_laila | 30 | 93.3% | no | ok (top confusions: safe -> violating, sensitive -> violating) |
| ann_mostafa | 30 | 76.7% | YES | accuracy 76.7% below floor 85%; recurring confusion sensitive -> safe (4 of 7 errors) (top confusions: sensitive -> safe, safe -> violating) |
| ann_sara | 30 | 100.0% | no | ok |

## PII screening

| Item | Type | Value |
|---|---|---|
| item-sensitive-10 | credit_card | `4111 1111 1111 1111 ` |
| item-sensitive-07 | phone | `0120 555 1234` |
| item-sensitive-03 | email | `mona.example@mail.com` |
| item-sensitive-14 | egyptian_national_id | `29012011234567` |

## Recommended next audit sample (15 items)

Disagreement items and flagged-annotator items first, then random fill:

`item-safe-04`, `item-safe-06`, `item-safe-08`, `item-safe-11`, `item-safe-13`, `item-safe-14`, `item-safe-15`, `item-safe-17`, `item-sensitive-04`, `item-sensitive-05`, `item-sensitive-06`, `item-sensitive-12`, `item-sensitive-20`, `item-violating-03`, `item-violating-16`
