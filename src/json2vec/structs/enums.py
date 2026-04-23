import enum


class Tokens(enum.IntEnum):
    valued = 0
    null = 1
    padded = 2
    masked = 3
    pruned = 4
    other = 5


class Stage(enum.StrEnum):
    fit = "fit"
    validate = "validate"
    test = "test"
    predict = "predict"


class Strata(enum.StrEnum):
    train = "train"
    validate = "validate"
    test = "test"
    predict = "predict"

    @classmethod
    def from_stage(cls, stage: Stage | str) -> list["Strata"]:
        match stage:
            case Stage.fit:
                return [cls.train, cls.validate]
            case Stage.validate:
                return [cls.validate]
            case Stage.test:
                return [cls.test]
            case Stage.predict:
                return [cls.predict]
            case _:
                raise ValueError(f"Unknown stage: {stage}")


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


class Component(enum.StrEnum):
    Request = "Request"
    Embedder = "Embedder"
    Decoder = "Decoder"
    TensorField = "TensorField"
    loss = "loss"
    write = "write"
    plot = "plot"
