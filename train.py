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


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path(args.data_root) if args.data_root else Path("/kaggle/input")
    print(f"[ENV] device={device} langs={args.langs}")
    work = Path("data_multiner"); work.mkdir(exist_ok=True)
    all_train_texts, all_train_ents, dev_pairs, eval_pairs = [], [], [], []
    for lang in args.langs:
        train_file, dev_file = find_files(base, lang)
        if train_file is None or dev_file is None:
            print(f"[LANG] missing conll for {lang}; skipping")
            continue
        tr = [to_io_pair(d) for d in read_conll_tsv(train_file)]
        dv = [to_io_pair(d) for d in read_conll_tsv(dev_file)]
        print(f"[LANG-{lang}] train={len(tr)} dev={len(dv)}")
        all_train_texts += [t for t, _ in tr]
        all_train_ents += [e for _, e in tr]
        dev_pairs.append((lang, dv))
        eval_pairs.append((lang, dev_file))
    to_docbin(all_train_texts, all_train_ents, work / "train.spacy")
    dev_all = []
    for lang, dv in dev_pairs:
        dev_all += [g for _, g in dv]
    to_docbin([t for t, _ in sum([list(dv) for _, dv in dev_pairs], [])], dev_all, work / "dev.spacy")
    nlp = train_multiner(work / "train.spacy", work / "dev.spacy", work / "multiner", epochs=args.epochs)
    per_lang = {}
    metrics_by_lang = {}
    for lang, dv in dev_pairs:
        texts = [t for t, _ in dv]
        golds = [list(g) for _, g in dv]
        pred = extract_ents(nlp, texts)
        rep = evaluate_ents(pred, golds)
        rep["detect_lang_counts"] = {l: sum(1 for t in texts if detect_language(t) == l) for l in ("en", "fr", "ar", "other")}
