"""Environment-configured ASGI entry point for multi-worker serving."""

from relflow.inference.deployment import Deployment

app = Deployment().app()
