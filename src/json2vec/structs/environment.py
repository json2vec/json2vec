from __future__ import annotations

import os
from typing import Literal, Self
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from json2vec.structs.enums import ShardingStrategy


class DataLoaderEnvironment(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    num_workers: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("JSON2VEC_NUM_WORKERS", "NUM_WORKERS"),
    )
    persistent_workers: bool = Field(
        default=True,
        validation_alias=AliasChoices("JSON2VEC_PERSISTENT_WORKERS", "PERSISTENT_WORKERS"),
    )
    pin_memory: bool = Field(
        default=True,
        validation_alias=AliasChoices("JSON2VEC_PIN_MEMORY", "PIN_MEMORY"),
    )
    sharding: ShardingStrategy = Field(
        default=ShardingStrategy.file,
        validation_alias=AliasChoices("JSON2VEC_SHARDING", "JSON2VEC_SHARDING_STRATEGY", "SHARDING_STRATEGY"),
    )
    chunk_batch_size: int = Field(
        default=4096,
        ge=1,
        validation_alias=AliasChoices("JSON2VEC_CHUNK_BATCH_SIZE", "JSON2VEC_PYARROW_BATCH_SIZE", "CHUNK_BATCH_SIZE"),
    )

    @field_validator("sharding", mode="before")
    @classmethod
    def normalize_sharding(cls, value: ShardingStrategy | str) -> ShardingStrategy | str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized

        return value

    @classmethod
    def from_env(cls) -> Self:
        return cls()


class TrackingEnvironment(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    logger: str | None = Field(default=None, validation_alias=AliasChoices("JSON2VEC_LOGGER"))
    wandb_api_key: str | None = Field(default=None, validation_alias=AliasChoices("WANDB_API_KEY"))
    neptune_api_token: str | None = Field(default=None, validation_alias=AliasChoices("NEPTUNE_API_TOKEN"))
    comet_api_key: str | None = Field(default=None, validation_alias=AliasChoices("COMET_API_KEY"))
    mlflow_tracking_uri: str | None = Field(default=None, validation_alias=AliasChoices("MLFLOW_TRACKING_URI"))
    tensorboard_log_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices("JSON2VEC_TENSORBOARD_LOG_DIR", "TENSORBOARD_LOG_DIR"),
    )
    csv_log_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices("JSON2VEC_CSV_LOG_DIR", "CSV_LOG_DIR"),
    )

    @field_validator("*", mode="before")
    @classmethod
    def strip_string_values(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            return stripped

        return value

    @property
    def resolved_tensorboard_log_dir(self) -> str:
        return self.tensorboard_log_dir or "logs/tensorboard"

    @property
    def resolved_csv_log_dir(self) -> str:
        return self.csv_log_dir or "logs/csv"

    @classmethod
    def from_env(cls) -> Self:
        return cls()


class DeploymentEnvironment(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    checkpoint: str = Field(
        default="model.ckpt",
        validation_alias=AliasChoices("JSON2VEC_CHECKPOINT", "CHECKPOINT"),
    )
    max_batch_size: int = Field(
        default=128,
        ge=1,
        validation_alias=AliasChoices("JSON2VEC_MAX_BATCH_SIZE", "MAX_BATCH_SIZE"),
    )
    batch_timeout: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("JSON2VEC_BATCH_TIMEOUT", "BATCH_TIMEOUT"),
    )
    workers_per_device: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("JSON2VEC_WORKERS_PER_DEVICE", "JSON2VEC_N_WORKERS", "N_WORKERS"),
    )
    accelerator: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        validation_alias=AliasChoices("JSON2VEC_ACCELERATOR", "ACCELERATOR"),
    )
    track_requests: bool = Field(
        default=False,
        validation_alias=AliasChoices("JSON2VEC_TRACK_REQUESTS", "TRACK_REQUESTS"),
    )

    @field_validator("checkpoint", "accelerator", mode="before")
    @classmethod
    def strip_required_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                raise ValueError(f"{info.field_name} must not be blank")
            return stripped

        return value

    @classmethod
    def from_env(cls) -> Self:
        return cls()