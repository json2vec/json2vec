import enum
from collections.abc import Mapping
from typing import TypeVar

T = TypeVar("T")


class Tokens(enum.IntEnum):
    valued = 0
    null = 1
    padded = 2
    masked = 3
    other = 4


class Strata(enum.StrEnum):
    train = "train"
    validate = "validate"
    test = "test"
    predict = "predict"

    @classmethod
    def normalize(cls, value: "Strata | str") -> "Strata":
        if isinstance(value, cls):
            return value

        return cls(str(value).strip().lower())

    @classmethod
    def expand(cls, value: T | Mapping["Strata | str", T], *, default: T) -> dict["Strata", T]:
        if isinstance(value, Mapping):
            normalized = {strata: default for strata in cls}
            for key, item in value.items():
                normalized[cls.normalize(key)] = item
            return normalized

        return {strata: value for strata in cls}


class Suffix(enum.StrEnum):
    feather = "feather"
    parquet = "parquet"
    ndjson = "ndjson"
    avro = "avro"
    csv = "csv"
    orc = "orc"
    json = "json"


class TensorKey(enum.StrEnum):
    value = "value"
    content = "content"
    state = "state"
    intervals = "intervals"
    probability = "probability"
    topk = "topk"
    embedding = "embedding"


class Metric(enum.StrEnum):
    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    loss = "loss"
    sigma = "sigma"
    throughput = "throughput"
    mae = "mae"
    rmse = "rmse"


class ShardingStrategy(enum.StrEnum):
    file = "file"
    chunk = "chunk"
    record = "record"

    @classmethod
    def normalize(cls, value: "ShardingStrategy | str") -> "ShardingStrategy":
        if isinstance(value, cls):
            return value

        return cls(value.strip().lower())

    @classmethod
    def expand(
        cls,
        value: "ShardingStrategy | str | Mapping[Strata | str, ShardingStrategy | str]",
        *,
        default: "ShardingStrategy",
    ) -> dict[Strata, "ShardingStrategy"]:
        return {
            strata: cls.normalize(strategy)
            for strata, strategy in Strata.expand(value, default=default).items()
        }


class AttentionMode(enum.StrEnum):
    mha = "mha"
    gqa = "gqa"
    mqa = "mqa"
    none = "none"

    @classmethod
    def normalize(cls, value: "AttentionMode | str") -> "AttentionMode":
        if isinstance(value, cls):
            return value

        return cls(value.strip().lower())

    def kv_heads(self, n_heads: int) -> int:
        match self:
            case AttentionMode.mha:
                return n_heads
            case AttentionMode.gqa:
                return max(1, n_heads // 2)
            case AttentionMode.mqa:
                return 1
            case AttentionMode.none:
                raise ValueError("attention mode 'none' does not define key/value heads")


class Component(enum.StrEnum):
    Request = "Request"
    Embedder = "Embedder"
    Decoder = "Decoder"
    TensorField = "TensorField"
    loss = "loss"
    write = "write"
    plot = "plot"
