"""
NLP Pipeline — Tokenizer & Vocabulary
--------------------------------------
Builds a word-level vocabulary from a corpus, encodes / decodes text,
and supports BOS / EOS special tokens for language-model use.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import List, Optional

import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Download NLTK resources once
for res in ("punkt", "punkt_tab", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

stemmer = EnglishStemmer()
lemmatizer = WordNetLemmatizer()

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Split text into NLTK word tokens"""
    return nltk.word_tokenize(text)


def normalize(token: str) -> str:
    """NFC → lower-case → lemmatise (or stem if unchanged)"""
    nfc = unicodedata.normalize("NFC", token).lower()
    lemma = lemmatizer.lemmatize(nfc)
    return lemma if lemma != nfc else stemmer.stem(nfc)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """
    Word-level tokenizer with a built-in vocabulary.

    Parameters
    ----------
    texts:
        Corpus used to build the vocabulary.
    min_count:
        Words that appear fewer than *min_count* times are mapped to <UNK>.
    normalize_tokens:
        Apply stemming/lemmatization during vocabulary construction.
    """

    def __init__(
        self,
        texts: List[str],
        min_count: int = 1,
        normalize_tokens: bool = False,
    ) -> None:
        self.min_count = min_count
        self.normalize_tokens = normalize_tokens
        self.word2idx: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word: List[str] = list(SPECIAL_TOKENS)
        self.word2count: Counter = Counter()
        self._build_vocabulary(texts)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _normalize(self, token: str) -> str:
        return normalize(token) if self.normalize_tokens else token.lower()

    def _build_vocabulary(self, texts: List[str]) -> None:
        for text in texts:
            for tok in tokenize(text):
                self.word2count[self._normalize(tok)] += 1
        for word, count in self.word2count.items():
            if count >= self.min_count and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode_word(self, token: str) -> int:
        return self.word2idx.get(self._normalize(token), UNK_ID)

    def encode(self, text: str) -> List[int]:
        return [self.encode_word(tok) for tok in tokenize(text)]

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        words: List[str] = []
        for idx in ids:
            w = self.idx2word[idx]
            if skip_special and w in SPECIAL_TOKENS:
                continue
            words.append(w)
        return " ".join(words)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, item: str) -> bool:
        return self._normalize(item) in self.word2idx

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={len(self)}, min_count={self.min_count})"


class BosEosTokenizer(Tokenizer):
    """Tokenizer that optionally prepends BOS and appends EOS."""

    def encode(  # type: ignore[override]
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        ids = super().encode(text)
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        return super().decode(ids, skip_special=skip_special)
