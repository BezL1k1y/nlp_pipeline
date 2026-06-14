# NLP Pipeline — from Scratch to Sentiment

A self-contained NLP pipeline built **without** high-level wrappers:

| Module | What it does |
|---|---|
| `tokenizer.py` | Word-level tokenizer with vocabulary, BOS/EOS support, stemming/lemmatisation |
| `tfidf.py` | Custom TF-IDF vectoriser with L2 normalisation |
| `ngram_lm.py` | N-gram conditional language model with greedy & sampled autocompletion |
| `demo.py` | End-to-end demo on Stanford IMDB |

## Quick start

```bash
pip install nltk scikit-learn datasets tqdm
python demo.py
```

## What runs

1. **Tokenizer smoke-test** — encodes / decodes a sentence, shows vocab size.
2. **TF-IDF + Logistic Regression** — trains on 5 000 IMDB reviews, evaluates on 2 000.  
   Custom accuracy is compared to sklearn's `TfidfVectorizer` baseline (both ≥ 0.80).
3. **Trigram LM** — autocompletes four movie-review prompts in greedy and sampled mode.

## Key design decisions

- **Normalisation pipeline**: NFC → lower-case → WordNet lemmatiser → Snowball stemmer (fallback).
- **IDF smoothing**: `idf(t) = -log((df(t) + 1) / N)` — avoids log(0) for unseen terms.
- **Temperature sampling** in the LM lets you trade diversity for coherence.
- `BosEosTokenizer` extends `Tokenizer` cleanly without duplicating vocabulary logic.

## Example output

```
Prompt  : 'the movie was'
  Greedy : the movie was a great film
  Sampled: the movie was not really that interesting at all
```
