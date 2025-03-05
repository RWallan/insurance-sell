from typing import Optional, Type

from imblearn.base import SamplerMixin
from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, TransformerMixin


class MinioSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str


class Settings(BaseSettings):
    DATA_SOURCES: list[str] = [
        'https://raw.githubusercontent.com/prsdm/mlops-project/refs/heads/main/data/train.csv',
        'https://raw.githubusercontent.com/prsdm/mlops-project/refs/heads/main/data/test.csv',
    ]
    DATA_SOURCES_BUCKET: str = 'raw'
    OUTPUT_DATA_BUCKET: str = 'raw.csv'


class ModelSelectionParams(BaseModel):
    model_selection: Type[MetaEstimatorMixin]
    params: dict

    @field_validator('model_selection', mode='before')
    @classmethod
    def validate_model_selection_class(cls, v):
        """Validate if is a valid scikitlearn model selection."""
        import importlib

        if v is None:
            return None

        module_name, class_name = v.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (AttributeError, ModuleNotFoundError):
            raise ValueError(f'Invalid model class: {v}')


class ModelParams(BaseModel):
    model: Type[ClassifierMixin]
    params: dict  # parametros do modelo

    @field_validator('model', mode='before')
    @classmethod
    def validate_model_class(cls, v):
        """Validate if is a valid scikitlearn classifier."""
        import importlib

        module_name, class_name = v.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (AttributeError, ModuleNotFoundError):
            raise ValueError(f'Invalid model class: {v}')


class Transformer(BaseModel):
    name: str
    transformer: Type[TransformerMixin]
    params: dict

    @field_validator('transformer', mode='before')
    @classmethod
    def validate_transformer_class(cls, v):
        """Validate if is a valid scikitlearn transformer."""
        import importlib

        module_name, class_name = v.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (AttributeError, ModuleNotFoundError):
            raise ValueError(f'Invalid model class: {v}')


class Resampler(BaseModel):
    name: str
    resampler: Type[SamplerMixin]
    params: dict

    @field_validator('resampler', mode='before')
    @classmethod
    def validate_transformer_class(cls, v):
        """Validate if is a valid scikitlearn resample."""
        import importlib

        module_name, class_name = v.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (AttributeError, ModuleNotFoundError):
            raise ValueError(f'Invalid model class: {v}')


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(toml_file='model_config.toml')
    model: ModelParams
    model_selection: Optional[ModelSelectionParams] = None
    resampler: Optional[Resampler] = None
    pipeline: list[Transformer]
    features: list[str]
    target: str

    @classmethod
    def settings_customise_sources(  # noqa: D102
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)
