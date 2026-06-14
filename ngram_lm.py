"""
N-Gram Language Model
---------------------
Builds an n-gram frequency model over a text corpus and supports
greedy / sampled autocompletion.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from tokenizer import BosEosTokenizer, EOS_ID, UNK_ID


class NGramLanguageModel:
    """
    Conditional n-gram language model.

    For each (n-1)-gram prefix, stores a Counter of observed continuations.
    Generation follows the prefix of length (n-1) from the last tokens.

    Parameters
    ----------
    n:
        Order of the n-gram model (must be ≥ 2).
    tokenizer:
        A fitted :class:`BosEosTokenizer`.
    texts:
        Training corpus.
    """

    def __init__(
        self,
        n: int,
        tokenizer: BosEosTokenizer,
        texts: List[str],
    ) -> None:
        if n < 2:
            raise ValueError("n must be ≥ 2")
        self.n = n
        self.tokenizer = tokenizer
        # prefix_tuple → Counter{continuation_id: count}
        self.frequencies: Dict[tuple, Counter] = defaultdict(Counter)
        self._build(texts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build(self, texts: List[str]) -> None:
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=False, add_eos=True)
            for i in range(len(ids)):
                prefix = tuple(ids[max(0, i - self.n + 1): i])
                self.frequencies[prefix][ids[i]] += 1

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_next_token(
        self,
        prefix: List[int],
        sample: bool = False,
        temperature: float = 1.0,
    ) -> int:
        """Return the most likely (or sampled) next token for prefix."""
        key = tuple(prefix[-(self.n - 1):]) if prefix else ()
        dist = self.frequencies.get(key, Counter())
        if not dist:
            return UNK_ID
        if sample:
            tokens, counts = zip(*dist.items())
            weights = [c ** (1.0 / temperature) for c in counts]
            return random.choices(tokens, weights=weights)[0]
        return dist.most_common(1)[0][0]

    def autocomplete(
        self,
        text: str,
        max_len: int = 32,
        skip_special: bool = True,
        sample: bool = False,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """
        Extend *text* until EOS or *max_len* tokens.

        Parameters
        ----------
        text:
            Seed text.
        max_len:
            Maximum total number of tokens in the output.
        skip_special:
            Whether to strip special tokens from the decoded string.
        sample:
            Sample from the distribution instead of taking the argmax.
        temperature:
            Softmax temperature (only used when *sample=True*).
        seed:
            Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        ids = self.tokenizer.encode(text, add_bos=not skip_special, add_eos=False)
        while len(ids) < max_len:
            prefix = ids[-(self.n - 1):] if len(ids) >= self.n - 1 else ids[:]
            next_id = self.generate_next_token(prefix, sample=sample, temperature=temperature)
            ids.append(next_id)
            if next_id == EOS_ID:
                break
        return self.tokenizer.decode(ids, skip_special=skip_special)
