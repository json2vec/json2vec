from typing import Annotated

import pydantic

from json2vec.inference.deployment import Deployment, DeploymentEnvironment

checkpoint = "examples/checkpoints/epoch=19-step=40.ckpt"

config = DeploymentEnvironment(checkpoint=checkpoint, max_batch_size=16, batch_timeout=0.0, workers_per_device=1, accelerator="mps")

class Request(pydantic.BaseModel):
    color: Annotated[str, pydantic.Field(pattern=r"^[A-Za-z]$")]


if __name__ == "__main__":
    Deployment.forge(request=Request).serve(config)
