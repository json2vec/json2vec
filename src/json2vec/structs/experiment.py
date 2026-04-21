import enum
import inspect
import json
import tomllib
from pathlib import Path
from typing import Annotated, Any, ClassVar, Iterable, Literal, Self, Type, TypeAlias, TypedDict, get_type_hints

import _jsonnet
import jsonpatch
import pydantic
import yaml
from faker import Faker

from json2vec.processors.base import PROCESSORS
from json2vec.structs.enums import Stage, Strata, Suffix
from json2vec.structs.structure import Structure
from json2vec.structs.tree import Address

fake = Faker()



def forge(cls: Type) -> TypeAlias:

    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)

    namespace: dict[str, Any] = {
        k: hints.get(k, v.annotation)
        for k, v in sig.parameters.items()
        if k != "self"
    }
    return TypedDict(f"{cls.__name__}Parameters", namespace, total=False)


class Dataset(pydantic.BaseModel):

    root: str | None = None
    sample_rate: Annotated[float, pydantic.Field(gt=0.0, le=1.0, default=1.0)]
    file_buffer_size: Annotated[int, pydantic.Field(gt=0)]
    observation_buffer_size: Annotated[int, pydantic.Field(gt=0)]
    processor: Annotated[str, pydantic.Field(default="default")]
    kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    suffix: Suffix
    patterns: dict[Strata, str]

    @pydantic.model_validator(mode="after")
    def check_processor_registered(self):

        if self.processor is not None and self.processor not in PROCESSORS:
            raise ValueError(f"you haven't registered processor {self.processor}")

        return self



class PatchOp(TypedDict, total=False):
    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str
    value: Any
    from_: str  # "from" is a keyword in Python


class ConfigSuffix(enum.StrEnum):
    jsonnet = ".jsonnet"
    yaml = ".yaml"
    yml = ".yml"
    toml = ".toml"
    json = ".json"


class Session(pydantic.BaseModel):

    name: str
    dataset: Dataset
    structure: Structure

    task: Stage
    trainer: dict[str, Any] = pydantic.Field(default_factory=dict)
    patience: Annotated[int, pydantic.Field(ge=1)] | None = None
    patches: list[PatchOp] = pydantic.Field(default_factory=list)

    pruned: list[Address] = pydantic.Field(default_factory=list)
    reset: list[Address] = pydantic.Field(default_factory=list)
    output: list[Address] = pydantic.Field(default_factory=list)

    learning_rate: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    weight_decay: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.01)]
    warmup_ratio: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.1)]
    min_lr_ratio: Annotated[float, pydantic.Field(gt=0.0, le=1.0, default=0.1)]
    p_prune: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)]
    p_mask: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)]

    @pydantic.model_validator(mode="after")
    def check_overriden_fields(self):

        for attribute in ["pruned", "reset"]:
            for field in getattr(self, attribute, []):
                if field not in self.structure.requests:
                    raise ValueError(f"{attribute} field '{field}' not found in structure requests")

        for attribute in ["output"]:
            for field in getattr(self, attribute, []):
                if field not in self.structure.contexts and field not in self.structure.requests:
                    raise ValueError(f"{attribute} context '{field}' not found in structure contexts or requests")

        return self

    @pydantic.model_validator(mode="after")
    def validate_trainer_epochs(self):

        min_epochs: Any = self.trainer.get("min_epochs")
        max_epochs: Any = self.trainer.get("max_epochs")

        for key, value in [("min_epochs", min_epochs), ("max_epochs", max_epochs)]:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"trainer.{key} must be an integer when provided")
            if value < 0:
                raise ValueError(f"trainer.{key} must be >= 0")

        if min_epochs is not None and max_epochs is not None and min_epochs > max_epochs:
            raise ValueError("trainer.min_epochs must be <= trainer.max_epochs")

        return self

    @pydantic.model_validator(mode="after")
    def validate_learning_rate_for_task(self):

        if self.task == Stage.fit and self.learning_rate is None:
            raise ValueError("learning_rate must be defined when task is 'fit'")

        if self.task != Stage.fit and self.learning_rate is not None:
            raise ValueError("learning_rate must not be defined when task is not 'fit'")

        return self

    @pydantic.model_validator(mode="after")
    def validate_patience_for_task(self):

        if self.task != Stage.fit and self.patience is not None:
            raise ValueError("patience must not be defined when task is not 'fit'")

        return self

    def patch(self, override: Iterable[PatchOp] | None = None, *, in_place: bool = True) -> Self:

        patches = list(self.patches if override is None else override)
        if not patches:
            return self

        data = self.model_dump(mode="python")
        patched = jsonpatch.apply_patch(data, patches, in_place=in_place)

        return self.__class__.model_validate(patched)



class Experiment(pydantic.BaseModel):
    _SUPPORTED_SUFFIXES: ClassVar[tuple[ConfigSuffix, ...]] = tuple(ConfigSuffix)

    project: str
    checkpoint: str|None = None

    name: Annotated[str, pydantic.Field(
        default_factory=lambda: f"{fake.word()}-{fake.color_name().lower()}"
    )]

    notes: Annotated[str, pydantic.Field(default="")]

    sessions: list[Session]

    @classmethod
    def _from_data(
        cls,
        data: dict[str, Any],
        *,
        name: str | None = None,
        notes: str | None = None,
    ) -> Self:
        experiment: Self = cls.model_validate(data)
        if name is not None:
            experiment.name = name

        if notes is not None:
            experiment.notes = notes

        return experiment

    @classmethod
    def from_yaml(
        cls,
        pathname: str | Path,
        *,
        name: str | None = None,
        notes: str | None = None,
    ) -> Self:
        path = Path(pathname)
        with path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle)
        return cls._from_data(data, name=name, notes=notes)

    @classmethod
    def from_toml(
        cls,
        pathname: str | Path,
        *,
        name: str | None = None,
        notes: str | None = None,
    ) -> Self:
        path = Path(pathname)
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls._from_data(data, name=name, notes=notes)

    @classmethod
    def from_json(
        cls,
        pathname: str | Path,
        *,
        name: str | None = None,
        notes: str | None = None,
    ) -> Self:
        path = Path(pathname)
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls._from_data(data, name=name, notes=notes)

    @classmethod
    def from_jsonnet(
        cls,
        pathname: str | Path,
        *,
        name: str | None = None,
        notes: str | None = None,
        ext_vars: dict[str, str] | None = None,
        tla_vars: dict[str, str] | None = None,
    ) -> Self:
        path = Path(pathname)
        rendered = _jsonnet.evaluate_file(
            str(path),
            ext_vars=ext_vars or {},
            tla_vars=tla_vars or {},
        )
        data: dict[str, Any] = json.loads(rendered)
        return cls._from_data(data, name=name, notes=notes)

    @classmethod
    def from_path(
        cls,
        pathname: str | Path,
        *,
        name: str | None = None,
        notes: str | None = None,
    ) -> Self:
        path = Path(pathname)
        suffix = path.suffix.lower()
        loaders: dict[ConfigSuffix, Any] = {
            ConfigSuffix.yaml: cls.from_yaml,
            ConfigSuffix.yml: cls.from_yaml,
            ConfigSuffix.toml: cls.from_toml,
            ConfigSuffix.json: cls.from_json,
            ConfigSuffix.jsonnet: cls.from_jsonnet,
        }

        try:
            config_suffix = ConfigSuffix(suffix)
        except ValueError:
            supported = ", ".join(suffix.value for suffix in cls._SUPPORTED_SUFFIXES)
            raise ValueError(f"Unsupported config suffix `{suffix}` for `{path}`. Supported suffixes: {supported}.")

        return loaders[config_suffix](path, name=name, notes=notes)

    @classmethod
    def _discover_experiments(cls, directory: Path) -> list[str]:
        supported: set[str] = {suffix.value for suffix in cls._SUPPORTED_SUFFIXES}
        stems: set[str] = {
            file.stem
            for file in directory.iterdir()
            if file.is_file() and file.suffix.lower() in supported
        }
        return sorted(stems)

    @classmethod
    def _resolve_experiment_path(cls, directory: Path, experiment: str) -> Path:
        target = Path(experiment)

        if target.suffix:
            candidate = target if target.is_absolute() else directory / target
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Experiment config not found: {experiment}")

        for suffix in cls._SUPPORTED_SUFFIXES:
            candidate = directory / f"{experiment}{suffix.value}"
            if candidate.exists():
                return candidate

        supported = ", ".join(suffix.value for suffix in cls._SUPPORTED_SUFFIXES)
        raise FileNotFoundError(
            f"Experiment `{experiment}` not found in `{directory}` with supported suffixes: {supported}"
        )

    @classmethod
    def from_config(
        cls,
        pathname: str | Path,
        experiment: str | None = None,
        name: str | None = None,
        notes: str | None = None,
    ) -> Self:
        directory = Path(pathname)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Experiment directory does not exist: {directory}")

        if experiment is None:
            experiments = cls._discover_experiments(directory)
            if not experiments:
                raise FileNotFoundError(f"No experiment files found under `{directory}`.")

            if len(experiments) == 1:
                experiment = experiments[0]
            else:
                options = ", ".join(experiments)
                raise ValueError(
                    "Experiment name is required when multiple configs exist under "
                    f"`{directory}`. Available experiments: {options}"
                )

        path = cls._resolve_experiment_path(directory, experiment)
        return cls.from_path(path, name=name, notes=notes)
