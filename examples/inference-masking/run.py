from __future__ import annotations

import pyarrow as pa
from rich import print

import relflow as rf


def training() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "letters": [
                    {"letter": "A", "next_letter": "B"},
                    {"letter": "B", "next_letter": "C"},
                    {"letter": "C", "next_letter": "D"},
                ]
            }
        ]
    )


def requests() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "letters": [
                    {"letter": "A"},
                    {"letter": "B"},
                    {"letter": "C"},
                ]
            }
        ]
    )


if __name__ == "__main__":
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        letters=rf.Branch(
            length=8,
            letter=rf.Category(size=26),
            next_letter=rf.Category(size=26, mask=True, topk=(3,)),
        ),
    )

    model.encode(training(), strata="train")
    predictions = model.predict(requests())

    print("The branch supplies cardinality; next_letter is source-less:")
    print(predictions["predictions"].to_pylist())
