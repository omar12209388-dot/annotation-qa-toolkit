"""PII flagging for annotation batches.

Pattern-based detection of common PII types in annotated text, with
Arabic-aware extras (Arabic-Indic digits in phone/ID patterns). This is
a screening layer for QA review, not a guarantee of PII-free data:
patterns catch common shapes, and a human reviewer adjudicates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Eastern Arabic-Indic digits are folded to Western digits before matching.
_EASTERN = "٠١٢٣٤٥٦٧٨٩"


def _fold_digits(text: str) -> str:
    return text.translate({ord(ch): str(i) for i, ch in enumerate(_EASTERN)})


_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"),
    "egyptian_national_id": re.compile(r"(?<!\d)[23]\d{13}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url_with_credentials": re.compile(r"https?://[^/\s:]+:[^@\s]+@"),
}


@dataclass
class PiiMatch:
    kind: str
    value: str
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_pii(text: str) -> list[PiiMatch]:
    """Return PII matches in the text, credit cards Luhn-validated."""
    folded = _fold_digits(text)
    matches: list[PiiMatch] = []
    for kind, pattern in _PATTERNS.items():
        for m in pattern.finditer(folded):
            if kind == "credit_card":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 16 and _luhn_ok(digits)):
                    continue
            matches.append(PiiMatch(kind, m.group(), m.start(), m.end()))
    # A phone-shaped run inside a stronger match (card, national ID) is
    # reported once, under the stronger kind.
    strong = [m for m in matches if m.kind != "phone"]
    matches = [
        m for m in matches
        if m.kind != "phone"
        or not any(s.start <= m.start and m.end <= s.end for s in strong)
    ]
    return matches


def redact(text: str, matches: list[PiiMatch] | None = None) -> str:
    """Replace matches with [KIND] placeholders, right to left."""
    if matches is None:
        matches = find_pii(text)
    folded = _fold_digits(text)
    for m in sorted(matches, key=lambda x: x.start, reverse=True):
        folded = folded[: m.start] + f"[{m.kind.upper()}]" + folded[m.end :]
    return folded
