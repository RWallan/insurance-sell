from typing import Optional, Type

import mlflow
import mlflow.models
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from prefect import flow, task
from prefect.logging import get_run_logger
from sklearn import metrics
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.model_selection._search import BaseSearchCV

from insurance_sell.settings import ModelSettings

TAGS = ['modeling', 'fit']


class Trainer:
    _transformers: list[tuple] = []
    _model: ClassifierMixin | BaseSearchCV
    _model_metrics: dict

    def __init__(self, settings: ModelSettings, run_id):
        """Class to handle fit pipeline.

        Args:
            settings: Model settings
            run_id: MlFlow run id
        """
        self._settings = settings
        self._transformers = self.create_pipeline()  # type: ignore
        self._model = self.configure_model()  # type: ignore
        self.run_id = run_id

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
        grid_params = next(
            (
                value
                for key, value in self._settings.model_selection.params.items()
                if key.startswith('param_')
            )
        )

        return self._settings.model_selection.model(
            model,  # type: ignore
            grid_params,  # type: ignore
            **first_level_params,
        )

    def report_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: pd.Series | np.ndarray,
        cohort: float,
        prefix: str = '',
    ):
        """Generate metrics for model.

        The metrics are:
            - Accuracy
            - AUC
            - Precision
            - Recall

        Args:
            y_true: Array with the true target classes
            y_proba: Array with the probability predicted
            cohort: Cohort to classify the class
            prefix: Metric prefix
        """
        y_pred = (y_proba[:, 1] > cohort).astype(int)
        acc = metrics.accuracy_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        return {
            f'{prefix}Accuracy': acc,
            f'{prefix}ROC AUC': auc,
            f'{prefix}Precision': precision,
            f'{prefix}Recall': recall,
        }

    @task(name='evaluate-model', tags=TAGS)
    def evalute_model(  # noqa: PLR0913
        self,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_test: pd.DataFrame,  # noqa: N803
        y_test: pd.Series,
        cohort: float = 0.5,
    ):
        """Evaluate model with train and test data.

        See `report_metrics` for details.

        Args:
            X_train: Train data
            y_train: Train target classes
            X_test: Test data
            y_test: Test target classes
            model: Model
            cohort: Cohort to classify the class.
        """
        _logger = get_run_logger()
        # FIX: Must verify if model is fitted before predict
        try:
            y_train_proba = self.model.predict_proba(X_train)
            y_test_proba = self.model.predict_proba(X_test)
        except AttributeError:
            _logger.error('Tried to evaluate model without fit')
            return

        train_metrics = self.report_metrics(
            y_train, y_train_proba, cohort, prefix='train_'
        )
        mlflow.log_metrics(train_metrics, run_id=self.run_id)
        test_metrics = self.report_metrics(
            y_test, y_test_proba, cohort, prefix='test_'
        )
        mlflow.log_metrics(test_metrics, run_id=self.run_id)

        self._model_metrics = {
            'Train Metrics': train_metrics,
            'Test Metrics': test_metrics,
        }

    def get_choosed_params(self):
        """Get the choosed model params."""
        return self.model.steps[-1][1].best_params_

    @flow(name='fit-model-and-evaluate')
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        evaluate: bool = True,
        *,
        X_test: Optional[pd.DataFrame | np.ndarray] = None,
        y_test: Optional[pd.Series | np.ndarray] = None,
    ):
        """Fit the model.

        Args:
            X_train: Train data.
            y_train: Target data.
            evaluate: If True, in the end of fit, the model will be evaluated
            X_test: Test data. Needed if evaluate is True
            y_test: Test target. Needed if evaluate is True
        """
        _logger = get_run_logger()
        _logger.info('Fitting model...')
        steps = self._transformers.copy()

        if self._settings.resampler:
            resampler = (
                self._settings.resampler.name,
                self._settings.resampler.resampler(
                    **self._settings.resampler.params
                ),
            )
            steps.extend([resampler, ('model', self._model)])
        else:
            steps.extend([('model', self._model)])
        # TODO: Mock to run tests
        self.model = Pipeline(steps=steps)
        self.model.fit(X_train, y_train)

        choosed_parameters = self.get_choosed_params()
        _logger.info(f'Choosed params: {choosed_parameters}')
        mlflow.log_params(choosed_parameters, run_id=self.run_id)
        signature = mlflow.models.infer_signature(
            model_input=X_train, params=choosed_parameters
        )
        mlflow.sklearn.log_model(self.model, 'model', signature=signature)

        if evaluate:
            _logger.info('Evaluating model...')
            self.evalute_model(X_train, y_train, X_test, y_test)
