from collections import Counter


def evaluate_ents(doc_ents_list, gold_ents_list):
    """Exact-span match per doc. Returns micro + per-type P/R/F1."""
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for pred, gold in zip(doc_ents_list, gold_ents_list):
        pred_set = set(pred)
        gold_set = set(gold)
        tp_all = pred_set & gold_set
        fp_all = pred_set - gold_set
        fn_all = gold_set - pred_set
        for _, _, l in tp_all:
            tp[l] += 1
        for _, _, l in fp_all:
            fp[l] += 1
        for _, _, l in fn_all:
            fn[l] += 1
    per_type = {}
    total_tp = total_fp = total_fn = 0
    for label in sorted(set(tp) | set(fp) | set(fn)):
        t = tp[label]
        f_ = fp[label]
        n = fn[label]
        total_tp += t
        total_fp += f_
        total_fn += n
        p = t / (t + f_) if t + f_ else 0.0
        r = t / (t + n) if t + n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_type[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                           "tp": t, "fp": f_, "fn": n}
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "micro": {"precision": round(micro_p, 3), "recall": round(micro_r, 3), "f1": round(micro_f, 3)},
        "per_type": per_type,
    }


def extract_ents(nlp, texts, labels_offset=()):
    """Runs the pipeline, returns list of (start, end, label) tuples per doc."""
    out = []
    for text in texts:
        doc = nlp(text)
        out.append([(e.start_char, e.end_char, e.label_) for e in doc.ents])
    return out


def merge_hybrid(ner_ents_list, ruler_ents_list):
    """NER wins overlaps; ruler fills unmatched gaps (its strength: rare/strict formats)."""
    merged = []
    for ner, ruler in zip(ner_ents_list, ruler_ents_list):
        ents = list(ner)
        ner_spans = [(s, e) for s, e, _ in ner]
        for s, e, l in ruler:
            if not any(not (e <= rs or s >= re_) for rs, re_ in ner_spans):
                ents.append((s, e, l))
        ents.sort()
        merged.append(ents)
    return merged
