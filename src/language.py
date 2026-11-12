FRENCH_STOP = ("le", "la", "les", "de", "des", "du", "et", "est", "avec", "qui", "pour", "dans")
ARABIC_TOKEN_RE = (0x0600, 0x06FF)


def _chars_in_range(s, lo, hi):
    return sum(1 for c in s if lo <= ord(c) <= hi)


def detect_language(text):
    tokens = text.split()
    if not tokens:
        return "other"
    arabic_tokens = [t for t in tokens if _chars_in_range(t, 0x0600, 0x06FF) / max(1, len(t)) > 0.5]
    if len(arabic_tokens) > len(tokens) * 0.4:
        return "ar"
    fr = sum(any(w.lower() == s for s in FRENCH_STOP) for w in tokens)
    has_diacritic = any(c in "éèêàçâùûôœ" for c in text.lower())
    if has_diacritic or fr >= len(tokens) * 0.15:
        return "fr"
    return "en"
