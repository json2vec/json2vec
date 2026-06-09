from __future__ import annotations

import random
import string

from rich import print

import json2vec as j2v
from json2vec.structs.enums import Strata

ALPHABET = string.ascii_uppercase
ADDRESS = "record/words/letters/letter"


def consecutive_letters(rng: random.Random, *, min_length: int = 3, max_length: int = 8) -> list[dict[str, str]]:
    length = rng.randint(min_length, max_length)
    start = rng.randint(0, len(ALPHABET) - length)
    return [{"letter": value} for value in ALPHABET[start : start + length]]


def words(rng: random.Random, *, min_length: int = 2, max_length: int = 4) -> list[dict[str, list[dict[str, str]]]]:
    length = rng.randint(min_length, max_length)
    return [{"letters": consecutive_letters(rng)} for _ in range(length)]


def records(n: int, *, seed: int = 7) -> list[dict[str, list[dict[str, list[dict[str, str]]]]]]:
    rng = random.Random(seed)
    return [{"words": words(rng)} for _ in range(n)]


if __name__ == "__main__":
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Array(
                j2v.Category(name="letter", max_vocab_size=len(ALPHABET), p_unavailable=0.0),
                name="letters",
                max_length=8,
                mask=j2v.Mask(count=2, window=2),
            ),
            name="words",
            max_length=3,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    data = records(5)
    print(data[0])

    inputs = model.encode(data, strata=Strata.train, mask=False)
    print(f"encoded field: {ADDRESS}")
    print(inputs[ADDRESS])

    masked = model.encode(data, strata=Strata.train, mask=True)
    print(f"masked field: {ADDRESS}")
    print(masked[ADDRESS])
