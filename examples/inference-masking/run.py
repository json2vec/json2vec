from __future__ import annotations

import string

from rich import print

import json2vec as j2v
from json2vec.structs.enums import Strata

ALPHABET = string.ascii_uppercase
ADDRESS = "record/letters/letter"
MASK = "<MASK>"


def letter_record(values: str) -> dict[str, list[dict[str, str]]]:
    return {"letters": [{"letter": value} for value in values]}


if __name__ == "__main__":
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Category(
                "letter",
                max_vocab_size=len(ALPHABET),
                p_unavailable=0.0,
                topk=[3],
            ),
            name="letters",
            max_length=8,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    # Prime the vocabulary from normal training inputs. The reserved MASK
    # literal is not part of this data and will not be learned as a category.
    model.encode(
        [
            letter_record("ABCDEFGH"),
            letter_record("IJKLMNOP"),
            letter_record("QRSTUVWX"),
            letter_record("YZ"),
        ],
        strata=Strata.train,
        mask=False,
    )

    print("observed vocabulary")
    print(model.nodes[ADDRESS].embedder.vocab.snapshot())
    print()

    query = [
        {
            "letters": [
                {"letter": "A"},
                {"letter": MASK},
                {"letter": "C"},
                {"letter": MASK},
                {"letter": "E"},
            ]
        }
    ]

    inputs = model.encode(query, strata=Strata.predict)
    print(f"predict field: {ADDRESS}")
    print(inputs[ADDRESS])
    print()

    predictions = model.predict(query)
    masked_slots = [index for index, item in enumerate(query[0]["letters"]) if item["letter"] == MASK]

    print("prediction addresses")
    print([str(address) for address in predictions])
    print()

    print(f"decoded content values: {ADDRESS}")
    print(predictions[ADDRESS]["content"]["value"])
    print()

    print(f"decoded top-3 candidates for {MASK} slots: {ADDRESS}")
    topk = predictions[ADDRESS]["content"]["topk"][0]
    print({slot: topk[slot] for slot in masked_slots})
