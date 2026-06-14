"""
TF-IDF Vectoriser
-----------------
Custom implementation of Term-Frequency × Inverse-Document-Frequency with
L2 normalisation, plus a thin wrapper around scikit-learn's TfidfVectorizer
for benchmarking.
"""

from __future__ import annotations

import numpy as np
from typing import List

from tokenizer import Tokenizer, UNK_ID


class TFIDFVectorizer:
    """
    TF-IDF vectoriser tied to a :class:`Tokenizer`.

    IDF formula (smoothed):
        idf(t) = -log((df(t) + 1) / N)

    where df(t) is the number of documents containing term t, and N is
    the total number of training documents.

    Parameters
    ----------
    tokenizer:
        A fitted :class:`Tokenizer` instance.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.num_docs: int = 0
        self.df: List[int] = [0] * len(tokenizer)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _add_document(self, text: str) -> None:
        self.num_docs += 1
        ids = set(self.tokenizer.encode(text))
        for tid in ids:
            if tid != UNK_ID:
                self.df[tid] += 1

    def fit(self, texts: List[str]) -> None:
        for text in texts:
            self._add_document(text)

    # ------------------------------------------------------------------
    # IDF / prediction
    # ------------------------------------------------------------------

    def idf(self, token_id: int) -> float:
        return -np.log((self.df[token_id] + 1) / self.num_docs)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Return an L2-normalised TF-IDF matrix of shape (n_docs, vocab_size).
        """
        mat = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text)
            n_tokens = len(ids)
            for tid in ids:
                if tid != UNK_ID:
                    mat[i, tid] += 1.0
            # Convert raw counts to TF-IDF
            for tid in np.nonzero(mat[i])[0]:
                mat[i, tid] = (mat[i, tid] / n_tokens) * self.idf(tid)
            norm = np.linalg.norm(mat[i])
            mat[i] /= norm + 1e-8
        return mat

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
