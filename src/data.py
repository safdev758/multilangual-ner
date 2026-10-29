"""MultiCoNER TSV (IOB2) to spaCy spacy DocBin, plus a code-switch-aware blank-pipeline trainer
using tok2vec/ner, plus a tiny grouped-language eval."""
import json
import random
from pathlib import Path

import spacy


def read_conll_tsv(path: Path):
    """Yields (text, [(start, end, label), ...]) from TSV: TOKEN\tLABEL.
    Handles space-merged tokens with WhitespaceInclusive.
    """
    docs = []
    tokens, tags = [], []
