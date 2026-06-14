"""
nlp_pipeline — end-to-end demo
================================
Runs three mini-experiments on the Stanford IMDB dataset:

1. Custom Tokenizer  — builds vocab, encodes / decodes sentences.
2. TF-IDF + Logistic Regression — sentiment classification (accuracy > 0.75).
3. N-gram Language Model — autocompletes movie-review prompts.

Usage
-----
    pip install nltk scikit-learn datasets tqdm
    python demo.py
"""

from __future__ import annotations

import random

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTFIDF
from sklearn.metrics import accuracy_score

from tokenizer import BosEosTokenizer, Tokenizer
from tfidf import TFIDFVectorizer
from ngram_lm import NGramLanguageModel


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_imdb(n_train: int = 5_000, n_test: int = 1_000):
    ds = load_dataset("stanfordnlp/imdb")
    train = [ds["train"].shuffle(seed=11)[i] for i in range(n_train)]
    test  = [ds["test"].shuffle(seed=11)[i]  for i in range(n_test)]
    train_texts  = [s["text"]  for s in train]
    train_labels = [s["label"] for s in train]
    test_texts   = [s["text"]  for s in test]
    test_labels  = [s["label"] for s in test]
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Experiment 1 — Tokenizer
# ---------------------------------------------------------------------------

def demo_tokenizer():
    print("\n\nExperiment 1: Tokenizer")
    print("=" * 60)
    corpus = ["Hello, world!", "I love Python!", "Hello, Python", "NLP is cool"]
    tok = Tokenizer(corpus, min_count=1)
    sentence = "Hello, Python! I love NLP"
    ids = tok.encode(sentence)
    decoded = tok.decode(ids)
    print(f"Input   : {sentence}")
    print(f"Encoded : {ids}")
    print(f"Decoded : {decoded}")
    print(f"Vocab size: {len(tok)}")


# ---------------------------------------------------------------------------
# Experiment 2 — TF-IDF
# ---------------------------------------------------------------------------

def demo_tfidf(train_texts, train_labels, test_texts, test_labels):
    print("\n\nExperiment 2: TF-IDF + Logistic Regression")
    print("=" * 60)

    # Custom implementation
    tokenizer = Tokenizer(train_texts, min_count=5)
    vec = TFIDFVectorizer(tokenizer)
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    acc   = accuracy_score(test_labels, preds)
    print(f"Custom TF-IDF accuracy  : {acc:.4f}")

    # sklearn baseline
    sk = SklearnTFIDF(min_df=5)
    X_tr_sk = sk.fit_transform(train_texts)
    X_te_sk = sk.transform(test_texts)
    clf2 = LogisticRegression(max_iter=1000)
    clf2.fit(X_tr_sk, train_labels)
    acc2 = accuracy_score(test_labels, clf2.predict(X_te_sk))
    print(f"sklearn TF-IDF accuracy : {acc2:.4f}")


# ---------------------------------------------------------------------------
# Experiment 3 — N-gram language model
# ---------------------------------------------------------------------------

def demo_ngram(train_texts):
    print("\n\nExperiment 3: N-gram Language Model (trigram)")
    print("=" * 60)

    tok = BosEosTokenizer(train_texts[:2_000], min_count=3)
    lm  = NGramLanguageModel(3, tok, train_texts[:2_000])

    prompts = [
        "the movie was",
        "this film has",
        "i really enjoyed",
        "the acting was",
    ]
    random.seed(11)
    for prompt in prompts:
        greedy  = lm.autocomplete(prompt, max_len=15, sample=False)
        sampled = lm.autocomplete(prompt, max_len=15, sample=True, temperature=1.2, seed=42)
        print(f"\nPrompt  : {prompt!r}")
        print(f"  Greedy : {greedy}")
        print(f"  Sampled: {sampled}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_tokenizer()
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    demo_tfidf(train_texts, train_labels, test_texts, test_labels)
    demo_ngram(train_texts)
