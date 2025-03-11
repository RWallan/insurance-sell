from typing import Type

import mlflow
from prefect import task
from prefect.logging import get_run_logger
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.model_selection._search import BaseSearchCV

from insurance_sell.settings import ModelSettings

TAGS = ['modeling', 'fit']


class Trainer:
    _transformers: list[tuple] = []
    model: ClassifierMixin | BaseSearchCV

    def __init__(self, settings: ModelSettings):
        """Class to handle fit pipeline."""
        self._settings = settings
        self._transformers = self.create_pipeline()  # type: ignore
        self.model = self.configure_model()  # type: ignore

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

    @task(name='configure-model', tags=TAGS)
    def configure_model(
        self,
    ):
        """Configure the estimator model."""
        _logger = get_run_logger()

        model = self._settings.model.model(**self._settings.model.params)

        if not self._settings.model_selection:
            return model

        _logger.info(
            (
                f'Using {self._settings.model_selection.name} with '
                f'{self._settings.model_selection.params}'
            )
        )
        first_level_params = {
            key: value
            for key, value in self._settings.model_selection.params.items()
            if not key.startswith('param_')
        }
        grid_params = {
            key: value
            for key, value in self._settings.model_selection.params.items()
            if key.startswith('param_')
        }

        return self._settings.model_selection.model(
            model,  # type: ignore
            grid_params,  # type: ignore
            **first_level_params,
        )
