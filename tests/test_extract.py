import logging

import pandas as pd
import pytest
from pandas import testing
from prefect.logging.loggers import disable_run_logger

from insurance_sell.extract import Extractor
from insurance_sell.settings import Settings

extractor = Extractor(Settings())


def test_fetch_data(tmp_path):
    df = pd.DataFrame(
        {
            'id': ['1', '2'],
            'Gender': ['m', 'f'],
            'Age': [20.0, 25.0],
            'HasDrivingLicense': [1.0, 0.0],
            'RegionID': [1.0, 2.0],
            'Switch': [1.0, 0.0],
            'VehicleAge': ['<1 year', '>=1 year'],
            'PastAccident': ['yes', 'no'],
            'AnnualPremium': ['£123', '£456'],
            'SalesChannelID': ['1', '2'],
            'DaysSinceCreated': [1, 2],
            'Result': [0, 1],
        }
    )

    df.to_csv(tmp_path / 'test.csv', index=False)

    with disable_run_logger():
        extractor.fetch_data.fn(extractor, str(tmp_path / 'test.csv'))

    testing.assert_frame_equal(df, extractor._raw_data)


def test_fetch_wrong_type_must_raises():
    with pytest.raises(ValueError, match='Source must be a csv file'):
        with disable_run_logger():
            extractor.fetch_data.fn(extractor, 'wrong.txt')


def test_fetch_wrong_schema_must_raises(tmp_path, caplog):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.to_csv(tmp_path / 'test.csv', index=False)

    with caplog.at_level(logging.ERROR):
        with disable_run_logger():
            extractor.fetch_data.fn(extractor, str(tmp_path / 'test.csv'))

    assert 'Failed to validate data schema' in caplog.text
