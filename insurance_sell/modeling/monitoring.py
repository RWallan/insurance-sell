import tempfile
from datetime import datetime

import pandas as pd
from evidently.future.datasets import (
    BinaryClassification,
    DataDefinition,
    Dataset,
)
from evidently.future.presets import (
    ClassificationPreset,
    DataDriftPreset,
    DataSummaryPreset,
)
from evidently.future.report import Report
from pandas.api.types import is_object_dtype
from prefect import flow, get_run_logger, task

from insurance_sell.helpers.settings import ModelSettings
from insurance_sell.helpers.storage import (
    BucketClient,
    get_file_from_storage,
    send_file_to_storage,
)
from insurance_sell.helpers.utils import get_model

TAGS = ['modeling', 'monitoring']


class Monitor:
    reference: pd.DataFrame
    current: pd.DataFrame

    def __init__(
        self, model: str, client: BucketClient, settings: ModelSettings
    ) -> None:
        """Class to monitor model performance."""
        self._settings = settings
        self._client = client
        self.model = get_model(model or 'models:/insurance-sell@production')
        self._get_data()

    def _get_data(self):
        with tempfile.NamedTemporaryFile(suffix='.csv') as f:
            reference = get_file_from_storage(
                self._client,
                self._settings.bucket_name,
                f.name,
                prefix='train_',
            )
            if not isinstance(reference, pd.DataFrame):
                raise Exception('Something went wrong while getting data.')
            reference['prediction'] = self.model['model'].predict_proba(
                reference.iloc[:, :-1]
            )
            self.reference = reference

            current = get_file_from_storage(
                self._client,
                self._settings.bucket_name,
                f.name,
                prefix='test_',
            )
            if not isinstance(current, pd.DataFrame):
                raise Exception('Something went wrong while getting data.')
            current['prediction'] = self.model['model'].predict_proba(
                current.iloc[:, :-1]
            )
            self.current = current

    @task(name='generate-schema', tags=TAGS)
    def generate_schema(self):
        """Create the column mapping for the monitor."""
        _logger = get_run_logger()
        self._target = self._settings.target
        features = self._settings.features

        self.categorical_features = [
            col
            for col in self.reference[features].columns  # type: ignore
            if is_object_dtype(self.reference[col].dtype)  # type: ignore
        ]
        _logger.info(f'Categorical features: {self.categorical_features}')
        self.numerical_features = list(
            set(features) - set(self.categorical_features)
        )
        _logger.info(f'Numerical features: {self.categorical_features}')

        self.schema = DataDefinition(
            numerical_columns=self.numerical_features,
            categorical_columns=self.categorical_features,
            classification=[
                BinaryClassification(
                    target=self._target,
                    prediction_probas='prediction',
                )
            ],
        )

    @flow
    def generate(self):
        """Generate reports for model."""
        self.generate_schema()
        reference = Dataset.from_pandas(
            self.reference,
            data_definition=self.schema,
        )
        current = Dataset.from_pandas(
            self.current,
            data_definition=self.schema,
        )

        self.data_drift = Report(
            metrics=[
                DataDriftPreset(),
                DataSummaryPreset(),
                ClassificationPreset(),
            ],
            include_tests=True,
        )
        self.data_drift.run(
            reference_data=reference,
            current_data=current,
        )
        self.data_drift.save_html('test.html')

        with tempfile.NamedTemporaryFile(suffix='.html') as f:
            self.data_drift.save_html(f.name)

            send_file_to_storage(
                self._client,
                'reports',
                f.name,
                f'{datetime.now()}_data_drift.html',
            )
