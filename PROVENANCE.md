# Provenance — Multilingual NER (MultiCoNER-2)

| Artifact | Source | Proof path |
|----------|--------|------------|
| Kernel v9 (final) | Kaggle `multilingual-ner-v9`, GPU, dataset `cryptexcode/multiconer-2` | `output/run9/output/metrics.json`, kernel log |
| en micro-F1 0.498 / fr 0.516 | same kernel | metrics.json → `en.micro`/`fr.micro` |
| Fix chain v3→v9 | parent-name match → file-name match; comment-line skip; space-split conll fields; ord() range check; mkdir before metrics write; metrics_by_lang init | kernel logs run3–run9 |
| Git repo | local real-timestamp commits | `git log` |
| MLflow | `multilingual-ner_training` | local mlruns/ |

## CAUTIONS
- Local conll fixture used for parsing tests only; full data lived on Kaggle only.
- micro-F1 ~0.5 is the transformer-free baseline; do not present as SOTA — cite as tok2vec benchmark.
