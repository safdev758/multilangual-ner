FRENCH_STOP = ("le", "la", "les", "de", "des", "du", "et", "est", "avec", "qui", "pour", "dans")
ARABIC_TOKEN_RE = (0x0600, 0x06FF)


def _chars_in_range(s, lo, hi):
    return sum(1 for c in s if lo <= ord(c) <= hi)


def detect_language(text):
    tokens = text.split()
    if not tokens:
        return "other"
    arabic_tokens = [t for t in tokens if _chars_in_range(t, 0x0600, 0x06FF) / max(1, len(t)) > 0.5]
