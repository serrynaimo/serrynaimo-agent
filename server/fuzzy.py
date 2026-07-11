"""Shared fuzzy matching for the lookup tools (mail, files, notifications, memory).

Dictated queries routinely misspell names ("Quantas" for Qantas, "Kahler" for
Kähler), and every lookup tool matches literally — one wrong letter means zero
hits and a dead end for the model. This module gives each tool the same cheap
recovery, no index required:

- ``variants(term)``: substring stems that sidestep wrong leading/trailing
  letters — searching "antas" finds Qantas however the query mangled it.
- ``close(term, text)`` / ``best_ratio(term, text)``: difflib similarity of a
  term against the words of a text, so stem-widened candidates can be kept or
  dropped by how much they actually resemble what was asked for.

Tools use it uniformly: literal match first; on zero hits, retry with stems
and keep candidates scoring ``close()`` to the original terms, labelling the
result approximate so the model confirms rather than asserts.
"""

import re
import unicodedata
from difflib import SequenceMatcher

# Similarity floor for calling two words "the same, misspelled". 0.72 lets
# Quantas/Qantas (0.77) and Kahler/Kähler through but keeps random words out.
THRESHOLD = 0.72


def _fold(s: str) -> str:
    """Lowercase and strip accents so Kähler == Kahler."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", _fold(text))


def variants(term: str) -> list[str]:
    """Substring stems of a term that survive common misspellings.

    A wrong first letter, doubled letter, or mangled ending still leaves the
    middle of the word intact — so search the middle. Only terms of 5+ chars
    produce variants (shorter stems match everything). Stems keep length >= 4.
    """
    t = _fold(term).strip()
    out = []
    if len(t) >= 5:
        for stem in (t[1:], t[:-1], t[1:-1], t[2:] if len(t) >= 7 else ""):
            if len(stem) >= 4 and stem not in out and stem != t:
                out.append(stem)
    return out


def best_ratio(term: str, text: str) -> float:
    """Highest difflib similarity between `term` and any word of `text`."""
    t = _fold(term)
    if not t:
        return 0.0
    best = 0.0
    for w in _words(text):
        # Cheap length gate before the quadratic ratio.
        if abs(len(w) - len(t)) <= max(2, len(t) // 3):
            r = SequenceMatcher(None, t, w).ratio()
            if r > best:
                best = r
        elif t in w or w in t:
            best = max(best, 0.75)
    return best


def close(term: str, text: str, threshold: float = THRESHOLD) -> bool:
    """Does `text` contain the term literally or a word close enough to it?"""
    t = _fold(term)
    return bool(t) and (t in _fold(text) or best_ratio(term, text) >= threshold)


def any_close(terms: list[str], text: str, threshold: float = THRESHOLD) -> bool:
    return any(close(t, text, threshold) for t in terms)
