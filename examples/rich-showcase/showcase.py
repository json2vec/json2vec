from rich import print

import relflow as rf

if __name__ == "__main__":
    model = rf.Model(
        name="order",
        d_model=16,
        n_layers=1,
        n_heads=4,
        amount="rf.Number",
        line_items=rf.Branch(
            length=4,
            sku=rf.Category(size=32),
            quantity=rf.Number,
        ),
        returned=rf.Category(target=True, size=2),
    )

    print(model)
