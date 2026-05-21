from typing import Annotated

import pydantic

from json2vec.inference.deployment import Deployment


class Request(pydantic.BaseModel):
    color: Annotated[str, pydantic.Field(pattern=r"^[A-Za-z]$")]


if __name__ == "__main__":

    (
        Deployment(
            checkpoint="examples/checkpoints/epoch=19-step=40.ckpt",
            max_batch_size=16,
            batch_timeout=0.0,
        )
        .forge(request=Request)
        .serve()
    )
