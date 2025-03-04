from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError
from prefect import task
from prefect.logging import get_run_logger
from prefect.tasks import exponential_backoff
from insurance_sell.settings import Settings

from .schemas import RawInsuranceSell

TAGS = ['extract', 'etl']


class Extractor:
    _raw_data: pd.DataFrame

    def __init__(self, settings: Settings):
        """Class to handle extract tasks."""
        self._raw_data = pd.DataFrame()
        self._settings = settings

    @task(
        name='fetch-data-from-source',
        retries=3,
        retry_delay_seconds=exponential_backoff(backoff_factor=10),
        retry_jitter_factor=1,
        tags=TAGS,
    )
    def fetch_data(
        self,
        src: str,
    ):
        """Fetch csv file from a source.

        Args:
            src: source from a csv file.
        """
        _logger = get_run_logger()

        if src.split('.')[-1] != 'csv':
            _logger.error(f'Failed to check if is a csv file from {src}')
            raise ValueError('Source must be a csv file')

        df = pd.read_csv(src)
        try:
            validated_data = RawInsuranceSell(df)
            self._raw_data = pd.concat([self._raw_data, validated_data])  # type: ignore

        except SchemaError as exc:
            _logger.error(f'Failed to validate data schema. {exc}')
    )

