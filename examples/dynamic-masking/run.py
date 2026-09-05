from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from rich import print

import relflow as rf

EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("kind", pa.large_string()),
            pa.field("amount", pa.float64()),
        ]
    )
)
MASKED_EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("kind", pa.large_string()),
            pa.field("amount", pa.float64()),
            pa.field("mask_event", pa.bool_(), nullable=False),
        ]
    )
)


@rf.preprocess(requires=("events",), produces=("events",))
def refunds(batch: rf.Batch) -> rf.Batch:
    column = batch.data.schema.get_field_index("events")
    lists = batch.data["events"].combine_chunks()
    records = pc.list_flatten(lists)
    selected = pc.fill_null(pc.less(records.field("amount"), 0), False)
    fields = list(MASKED_EVENTS.value_type)
    records = pa.StructArray.from_arrays(
        [selected if field.name == "mask_event" else records.field(field.name) for field in fields],
        fields=fields,
    )
    offsets = pc.subtract(lists.offsets, lists.offsets[0])
    values = pa.LargeListArray.from_arrays(
        offsets,
        records,
        type=MASKED_EVENTS,
        mask=pc.is_null(lists),
    )
    return batch.replace(batch.data.set_column(column, "events", values))


def records() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "events": [
                    {"kind": "purchase", "amount": 25.0},
                    {"kind": "refund", "amount": -10.0},
                    {"kind": "purchase", "amount": 8.0},
                ]
            }
        ],
        schema=pa.schema([pa.field("events", EVENTS)]),
    )


if __name__ == "__main__":
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        events=rf.Branch(
            length=3,
            mask=rf.Mask(
                query="mask_event",
                skip=True,
                dropout=False,
            ),
            kind=rf.Category(size=16),
            amount=rf.Number,
        ),
    )

    encoded = model.encode(
        records(),
        preprocess=refunds,
        strata="predict",
    )

    print("One branch decision is shared by kind and amount:")
    print(encoded[rf.Address("record/events/kind")])
    print(encoded[rf.Address("record/events/amount")])
