# Multilingual NER (MultiCoNER-2)

Language detection + spaCy Transformer-free NER on MultiCoNER-2 (English + French), single joined DocBin training run.

## What it does
- **Dataset**: MultiCoNER-2 conll format (`token _ _ TAG`, 4 tab-separated fields, tag in last column), 16.7k en + 16.5k fr sentences, 33 entity types each.
- **Pipeline**: per-language tokenizer → merged `train.spacy`/`dev.spacy` DocBins → spaCy NER (ner + tok2vec) 10 epochs.
