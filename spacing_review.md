# Spacing review (5-rule framework) — Multilingual NER

## D+2
1. Why does the conll parser read `parts[-1]` as the tag (4-field format with `_` placeholders)?
2. What is `alignment_mode="expand"` doing in char_span alignment and when does it create false positives?
3. Why does the fr Station tag get F1 0.22 with recall 0.60 (fp blowup)?

## D+7
1. Diagnose the Politician/Station confusion in per_type tables; what lexical priors would fix it?
2. Ablation: train separate per-language models vs the joined DocBin — which wins and why?
3. Language-detect counts: 18 fr-labeled docs in en set — is that data noise or detector error?

## D+21 (construct)
1. Swap tok2vec for XLM-R transformer (same DocBin/eval layers) and report delta.
2. Add a language-adapter layer so one model serves en/fr/ar in a single runtime.
3. Two-sentence resume blurb with micro-F1 and per-type extremes.

## Pointers
- loader src/data.py read_conll_tsv; docbin src/train_ner.py to_docbin; eval src/evaluation.py.
