"""Registered stateless metric configurations."""

from relflow.metrics.base import METRICS, Metric, MetricPlugin, MetricRegistry, Trait, register, registry
from relflow.metrics.extensions import MAE, RMSE, Accuracy, AngularMAE, Precision, Recall, Specificity

__all__ = [
    "METRICS",
    "Accuracy",
    "AngularMAE",
    "MAE",
    "Metric",
    "MetricPlugin",
    "MetricRegistry",
    "Precision",
    "RMSE",
    "Recall",
    "Specificity",
    "Trait",
    "register",
    "registry",
]
