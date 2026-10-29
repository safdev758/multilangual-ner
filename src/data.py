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
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    docs.append((tokens, tags))
                    tokens, tags = [], []
                continue
            if line.startswith("#"):
                continue
            if "\t" in line:
                fields = line.split("\t")
                if len(fields) >= 4:
                    tok, tag = fields[0], fields[-1]
                else:
                    tok, tag = fields[0], fields[1]
            else:
                parts = line.split()
                tok, tag = parts[0], parts[-1]
