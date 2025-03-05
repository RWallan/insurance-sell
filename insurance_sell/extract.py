import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pandera.errors import SchemaError
from prefect import flow, task
from prefect.futures import wait
from prefect.logging import get_run_logger
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.tasks import exponential_backoff

from insurance_sell.settings import Settings
from insurance_sell.storage import (
    BucketClient,
    check_if_object_exists,
    get_file_from_storage,
    send_file_to_storage,
)

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

    @task(
        name='persist-in-storage',
        tags=TAGS,
        cache_key_fn=lambda context,
        parameters: f'{parameters["bucket_name"]}-{parameters["input_file"]}-{parameters["output_file"]}-{parameters["overwrite"]}',  # noqa: E501
    )
    def _persist_in_storage(
        self,
        client: BucketClient,
        bucket_name: str,
        input_file: str,
        output_file: str,
        *,
        overwrite: bool,
    ):
        """Persist a file to bucket storage verifying if overwrite or not."""
        _logger = get_run_logger()

        if not overwrite:
            df = get_file_from_storage(
                client, bucket_name, input_file, output_file
            )
            if isinstance(df, pd.DataFrame):
                df = pd.concat([df, self._raw_data])
                df.to_csv(input_file, index=False)
            else:
                _logger.info(
                    (
                        f'Nothing to load {output_file} '
                        f'from {bucket_name}. Overwriting...'
                    )
                )
                self._raw_data.to_csv(input_file, index=False)
        else:
            self._raw_data.to_csv(input_file, index=False)

        send_file_to_storage(client, bucket_name, input_file, output_file)

    @task(name='persist-in-local', tags=TAGS)
    def _persist_in_local(self, output_file: str, overwrite: bool):
        """Persist a file in local verifying if overwrite or not."""
        _logger = get_run_logger()

        if not overwrite:
            _logger.warning('Tried to append if no local file found')
            overwrite = True if Path(output_file).exists() else False

        self._raw_data.to_csv(
            output_file, index=False, mode='w' if overwrite else 'a'
        )
        _logger.info(f'Saved {output_file}')

    @flow(
        name='extract-data-and-persists',
        task_runner=ThreadPoolTaskRunner(max_workers=2),  # type: ignore
    )
    def extract(  # noqa: PLR0913
        self,
        output_file: str | Path,
        sources: Optional[list[str]] = None,
        client: Optional[Any] = None,
        bucket_name: Optional[str] = None,
        *,
        overwrite: bool = False,
        persist_in_storage: bool = True,
    ):
        """Execute all extract data flow.

        This pipeline will:
            - Extract data from source
            - Validate with InsuranceSell schema
            - Persists de data in storage or local

        Args:
            sources: A list of csv sources.
            client: A bucket client to use.
            bucket_name: Bucket name.
            output_file: Output file name. If `persist_in_storage` is False, need be a local path.
            overwrite: If True, will overwrite existing data.
            persist_in_storage: If data will persist in storage.
        """  # noqa: E501
        _logger = get_run_logger()

        futures = []
        sources = sources or self._settings.DATA_SOURCES
        for source in sources:
            _logger.info(f'Extracting data for {source.split("/")[-1]}')
            futures.append(self.fetch_data.submit(source))  # type: ignore

        has_object = False
        if persist_in_storage:
            bucket_name = bucket_name or self._settings.DATA_SOURCES_BUCKET
            if bucket_name and client:
                # Check if can append in existing file
                if not overwrite:
                    has_object = check_if_object_exists(
                        client, bucket_name, output_file
                    )
                    if not has_object:
                        _logger.warning(
                            (
                                f'{output_file} not found in {bucket_name}. '
                                'Overwriting'
                            )
                        )
                        overwrite = True

                # Create a temporary file to save in storage
                with tempfile.NamedTemporaryFile(suffix='.csv') as f:
                    wait(futures)
                    self._persist_in_storage(
                        client,
                        bucket_name,
                        f.name,
                        output_file,
                        overwrite=overwrite,
                    )
            else:
                _logger.warning(
                    (
                        'Tried to persists in storage but no bucket or '
                        'client provided. Trying to save locally'
                    )
                )
                wait(futures)
                self._persist_in_local(output_file, overwrite)
        else:
            wait(futures)
            self._persist_in_local(output_file, overwrite)
