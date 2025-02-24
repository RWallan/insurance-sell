import logging
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError

from .schemas import RawInsuranceSell

DATA_PATH = Path().cwd() / 'data'
RANDOM_STATE = 13

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_data(overwrite: bool = False):
    """Extract data from https://github.com/prsdm/mlops-project/tree/main and export to csv.

    Args:
        overwrite: If True, the data extracted will overwrite existing data.
    """  # noqa: E501
    DATA_PATH.mkdir(exist_ok=True, parents=True)
    RAW_FILE = DATA_PATH / 'raw.csv'
    raw_data = pd.DataFrame()

    append_mode = True if RAW_FILE.exists() and not overwrite else False

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
        RAW_FILE,
        mode='a' if append_mode else 'w',
        header=not append_mode,
        index=False,
    )
    logger.info(
        'Raw data created at: %s.\nAppend mode: %s',
        str(RAW_FILE),
        append_mode,
    )


if __name__ == '__main__':
    extract_data()
