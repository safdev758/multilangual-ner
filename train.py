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
