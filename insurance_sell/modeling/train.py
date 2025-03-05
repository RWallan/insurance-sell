from typing import Type

import mlflow
from prefect import task
from prefect.logging import get_run_logger
from sklearn.base import TransformerMixin

from insurance_sell.settings import ModelSettings

TAGS = ['modeling', 'fit']


class Trainer:
    _transformers: list[tuple] = []

    def __init__(self, settings: ModelSettings):
        """Class to handle fit pipeline."""
        self._settings = settings
        self._transformers = self.create_pipeline()  # type: ignore

    @task(name='create-pipeline', tags=TAGS)
    def create_pipeline(self) -> list[tuple[str, Type[TransformerMixin]]]:
        """Create transform Pipeline."""
        _logger = get_run_logger()

        transformers = []
        for transformer in self._settings.pipeline:
            _logger.info(
                (f'{transformer.name}: using parameters {transformer.params}')
            )
            transformers.append(
                (
                    transformer.name,
                    transformer.transformer(**transformer.params),
                )
            )
            mlflow.set_tag('preprocessing', transformer.name)

        return transformers
