# Multilingual NER (MultiCoNER-2)

Language detection + spaCy Transformer-free NER on MultiCoNER-2 (English + French), single joined DocBin training run.

## What it does
- **Dataset**: MultiCoNER-2 conll format (`token _ _ TAG`, 4 tab-separated fields, tag in last column), 16.7k en + 16.5k fr sentences, 33 entity types each.
- **Pipeline**: per-language tokenizer → merged `train.spacy`/`dev.spacy` DocBins → spaCy NER (ner + tok2vec) 10 epochs.
- **Language detection**: rule-based (`src/language.py` — Arabic Unicode-range token ratio, French stopword/diacritic heuristics) logged as telemetry in metrics.

## Result (kernel `multilingual-ner-v9`)
| lang | micro precision | recall | F1 |
|------|-----------------|--------|----|
| en | 0.538 | 0.463 | **0.498** |
| fr | 0.537 | 0.498 | **0.516** |

Strongest types: PrivateCorp (fr F1 0.846), SportsGRP (0.72), HumanSettlement (0.73/0.64), Artist (0.66/0.64). Weakest: Politician, Station (fp-heavy), Food/ArtWork (tiny support).

Full per-type tables: `output/run9/output/metrics.json`.

## Layout
```
src/data.py          read_conll_tsv, to_io_tuple, tokens_with_offsets
src/train_ner.py     DocBin build + spaCy NER training
