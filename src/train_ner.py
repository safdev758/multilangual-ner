"""Train a blank-based multilingual NER over MultiCoNER 2 (en/fr/mi subsets).

Uses spaCy CLI train (tok2vec+ner), mirroring project 8's kernel pattern, with a
language column emitted alongside predictions for code-switching docs.
"""
import argparse
import json
from pathlib import Path

import spacy
from spacy.util import Config
from spacy.cli.init_config import init_config
from spacy.cli.train import train as spacy_train

from .data import read_conll_tsv, tokens_with_offsets
from .language import detect_language


def parse_args():
    p = argparse.ArgumentParser(description="Multilingual NER (MultiCoNER, en+fr to start)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--langs", type=str, nargs="+", default=["en", "fr"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--out", type=str, default="output")
    return p.parse_args()


def find_conll_dir(base: Path, lang: str):
    for p in base.rglob("**/train/*.tsv"):
        name = p.parent.name.lower()
        if name.startswith(lang + "_"):
            return p.parent
    for p in base.rglob("*.tsv"):
        pass
    return None


def find_train_dev(base: Path, lang: str):
    files = sorted(base.rglob("*.conll"))
    train = next(p for p in files if lang == p.parent.name and "train" in p.name)
    dev = next(p for p in files if lang == p.parent.name and "dev" in p.name)
    return train, dev


def to_docbin(texts, ents, out_path: Path):
    from spacy.tokens import Doc, DocBin

    nlp = spacy.blank("en")
    db = DocBin(store_user_data=True)
    n_with_ents = 0
    for text, ents_i in zip(texts, ents):
        doc = Doc(nlp.spacy_vocab if False else nlp.vocab, words=text.split(" "))
        spacy_ents = []
        for s, e, l in ents_i:
            span = doc.char_span(s, e, alignment_mode="expand", label=l)
