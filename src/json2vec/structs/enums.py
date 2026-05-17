import enum


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
