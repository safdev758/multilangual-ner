import argparse
import json
from pathlib import Path

import spacy
import torch

from src.data import read_conll_tsv, tokens_with_offsets
from src.language import detect_language
from src.train_ner import to_docbin, train_multiner
from src.evaluation import evaluate_ents, extract_ents


def parse_args():
    p = argparse.ArgumentParser(description="Multilingual NER (MultiCoNER, en+fr)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--langs", type=str, nargs="+", default=["en", "fr"])
    p.add_argument("--epochs", type=int, default=12)
    return p.parse_args()


def find_files(base, lang):
    files = sorted(base.rglob("*.conll"))
    train = next((p for p in files if p.name.startswith(lang) and "train" in p.name), None)
    dev = next((p for p in files if p.name.startswith(lang) and "dev" in p.name), None)
    if train is None:
        train = next((p for p in files if f"{lang}" in p.name and "train" in p.name), None)
    if dev is None:
        dev = next((p for p in files if f"{lang}" in p.name and "dev" in p.name), None)
    return train, dev


