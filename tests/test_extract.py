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


def test_persist_in_storage_must_append(client, tmp_path, monkeypatch):
    expected_appended_df_length = 4
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

    def mock_get_file_from_storage(*args):
        return df

    def mock_put_file_in_storage(*args):
        pass

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    with disable_run_logger():
        extractor._persist_in_storage.fn(
            extractor,
            client,
            'test',
            str(tmp_path / 'test.csv'),
            'output.csv',
            overwrite=False,
        )

    result_df = pd.read_csv(tmp_path / 'test.csv')

    assert len(result_df) == expected_appended_df_length


def test_persist_in_storage_must_overwrite(client, tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            'id': [1, 2],
            'Gender': ['m', 'f'],
            'Age': [20.0, 25.0],
            'HasDrivingLicense': [1.0, 0.0],
            'RegionID': [1.0, 2.0],
            'Switch': [1.0, 0.0],
            'VehicleAge': ['<1 year', '>=1 year'],
            'PastAccident': ['yes', 'no'],
            'AnnualPremium': ['£123', '£456'],
            'SalesChannelID': [1, 2],
            'DaysSinceCreated': [1, 2],
            'Result': [0, 1],
        }
    )

    df.to_csv(tmp_path / 'test.csv', index=False)

    def mock_get_file_from_storage(*args):
        return df

    def mock_put_file_in_storage(*args):
        pass

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    with disable_run_logger():
        extractor._persist_in_storage.fn(
            extractor,
            client,
            'test',
            str(tmp_path / 'test.csv'),
            'output.csv',
            overwrite=True,
        )

    result_df = pd.read_csv(tmp_path / 'test.csv')

    testing.assert_frame_equal(result_df, df)


def test_persist_in_storage_must_overwrite_if_dont_have_past_object(
    client, tmp_path, monkeypatch, caplog
):
    df = pd.DataFrame(
        {
            'id': [1, 2],
            'Gender': ['m', 'f'],
            'Age': [20.0, 25.0],
            'HasDrivingLicense': [1.0, 0.0],
            'RegionID': [1.0, 2.0],
            'Switch': [1.0, 0.0],
            'VehicleAge': ['<1 year', '>=1 year'],
            'PastAccident': ['yes', 'no'],
            'AnnualPremium': ['£123', '£456'],
            'SalesChannelID': [1, 2],
            'DaysSinceCreated': [1, 2],
            'Result': [0, 1],
        }
    )

    df.to_csv(tmp_path / 'test.csv', index=False)

    def mock_get_file_from_storage(*args):
        return None

    def mock_put_file_in_storage(*args):
        pass

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    with caplog.at_level(logging.ERROR):
        with disable_run_logger():
            extractor._persist_in_storage.fn(
                extractor,
                client,
                'test',
                str(tmp_path / 'test.csv'),
                'output.csv',
                overwrite=False,
            )

    result_df = pd.read_csv(tmp_path / 'test.csv')

    assert 'Error to load output.csv from test. Overwriting...' in caplog.text
    testing.assert_frame_equal(result_df, df)
