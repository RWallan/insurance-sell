from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError
from prefect import task
from prefect.logging import get_run_logger

from .schemas import RawInsuranceSell


@task
def extract_data(filename: Path | str, overwrite: bool = False):
    """Extract data from https://github.com/prsdm/mlops-project/tree/main and export to csv.

    Args:
        filename: File name to save data.
        overwrite: If True, the data extracted will overwrite existing data.
    """  # noqa: E501
    logger = get_run_logger()
    raw_data = pd.DataFrame()

    fname = Path(filename)

    train_data = pd.read_csv(
        'https://raw.githubusercontent.com/prsdm/mlops-project/refs/heads/main/data/train.csv',
    )

    try:
        train_data = RawInsuranceSell(train_data)
        raw_data = pd.concat([raw_data, train_data])  # type: ignore
    except SchemaError as exc:
        logger.error(f'Failed to validate `train` data schema. {exc}')

    test_data = pd.read_csv(
        'https://raw.githubusercontent.com/prsdm/mlops-project/refs/heads/main/data/test.csv',
    )

    try:
        test_data = RawInsuranceSell(test_data)
        raw_data = pd.concat([raw_data, test_data])  # type: ignore
    except SchemaError as exc:
        logger.error(f'Failed to validate `test` data schema. {exc}')

    if raw_data.empty:
        logger.error('Raw data not created. DataFrame is empty')
        return

    raw_data.to_csv(
        fname,
        mode='a' if not overwrite else 'w',
        header=overwrite,
        index=False,
    )
    logger.info(
        'Raw data created at: %s.\nAppend mode: %s',
        str(fname),
        not overwrite,
    )

    return fname
