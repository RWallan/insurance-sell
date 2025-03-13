import logging
from time import sleep

import pandas as pd
import pytest
from pandas import testing
from prefect.logging.loggers import disable_run_logger

from insurance_sell.extract import Extractor
from insurance_sell.helpers.settings import Settings


def test_fetch_data(tmp_path):
    extractor = Extractor(Settings())
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

    with disable_run_logger():
        extractor.fetch_data.fn(extractor, str(tmp_path / 'test.csv'))

    testing.assert_frame_equal(df, extractor._raw_data)


def test_fetch_wrong_type_must_raises():
    extractor = Extractor(Settings())
    with pytest.raises(ValueError, match='Source must be a csv file'):
        with disable_run_logger():
            extractor.fetch_data.fn(extractor, 'wrong.txt')


def test_fetch_wrong_schema_must_raises(tmp_path, caplog):
    extractor = Extractor(Settings())
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.to_csv(tmp_path / 'test.csv', index=False)

    with caplog.at_level(logging.ERROR):
        with disable_run_logger():
            extractor.fetch_data.fn(extractor, str(tmp_path / 'test.csv'))

    assert 'Failed to validate data schema' in caplog.text


def test_persist_in_storage_must_append(client, tmp_path, monkeypatch):
    extractor = Extractor(Settings())
    expected_appended_df_length = 4
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
    extractor._raw_data = df

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
    extractor = Extractor(Settings())
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
    extractor._raw_data = df

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
    extractor = Extractor(Settings())
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
    extractor._raw_data = df

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

    with caplog.at_level(logging.INFO):
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

    assert (
        'Nothing to load output.csv from test. Overwriting...' in caplog.text
    )
    testing.assert_frame_equal(result_df, df)


def test_extract_flow_must_overwrite_and_send_to_storage(
    client, tmp_path, monkeypatch
):
    expected_len_data = 200_000
    extractor = Extractor(Settings())

    def mock_get_file_from_storage(*args):
        pass

    def mock_put_file_in_storage(*args):
        temp = pd.read_csv(args[2])
        temp.to_csv(tmp_path / 'output.csv', index=False)

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    extractor.extract(
        'output.csv',
        client=client,
        bucket_name='test',
        overwrite=True,
    )

    result = pd.read_csv(tmp_path / 'output.csv')

    assert len(result) == expected_len_data


def test_extract_flow_must_append_file_and_send_to_storage(
    client, tmp_path, monkeypatch
):
    expected_len_data = 200_002
    extractor = Extractor(Settings())
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

    def mock_get_file_from_storage(*args):
        return df

    def mock_put_file_in_storage(*args):
        temp = pd.read_csv(args[2])
        temp.to_csv(tmp_path / 'output.csv', index=False)

    def mock_check_file(*args):
        return True

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.check_if_object_exists',
        mock_check_file,
    )

    extractor.extract(
        'output.csv',
        client=client,
        bucket_name='test',
        overwrite=False,
    )

    result = pd.read_csv(tmp_path / 'output.csv')

    assert len(result) == expected_len_data


def test_extract_flow_must_overwrite_file_and_send_to_storage_if_not_check(
    client, tmp_path, monkeypatch, caplog
):
    expected_len_data = 200_000
    extractor = Extractor(Settings())
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

    def mock_get_file_from_storage(*args):
        return df

    def mock_put_file_in_storage(*args):
        temp = pd.read_csv(args[2])
        temp.to_csv(tmp_path / 'output.csv', index=False)

    def mock_check_file(*args):
        return False

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.check_if_object_exists',
        mock_check_file,
    )

    with caplog.at_level(logging.WARNING):
        extractor.extract(
            'output.csv',
            client=client,
            bucket_name='test',
            overwrite=False,
        )

    result = pd.read_csv(tmp_path / 'output.csv')

    assert len(result) == expected_len_data
    assert 'output.csv not found in test. Overwriting' in caplog.text


def test_extract_flow_must_overwrite_file_and_save_local_if_bucket_vars_not_provided(  # noqa: E501
    client, tmp_path, monkeypatch, caplog
):
    expected_len_data = 200_000
    extractor = Extractor(Settings())
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

    def mock_get_file_from_storage(*args):
        return df

    def mock_put_file_in_storage(*args):
        temp = pd.read_csv(args[2])
        temp.to_csv(tmp_path / 'output.csv', index=False)

    def mock_check_file(*args):
        return False

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.check_if_object_exists',
        mock_check_file,
    )

    with caplog.at_level(logging.WARNING):
        extractor.extract(
            tmp_path / 'output.csv',
            overwrite=False,
        )

    # HACK: persist_in_local is submited, so read_csv will get data
    # before saved
    sleep(0.5)
    result = pd.read_csv(tmp_path / 'output.csv')

    assert len(result) == expected_len_data
    assert (
        'Tried to persists in storage but no bucket or '
        'client provided. Trying to save locally'
    ) in caplog.text


def test_extract_flow_must_save_local_if_not_persist_in_storage(  # noqa: E501
    client, tmp_path, monkeypatch, caplog
):
    expected_len_data = 200_000
    extractor = Extractor(Settings())
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

    def mock_get_file_from_storage(*args):
        return df

    def mock_put_file_in_storage(*args):
        temp = pd.read_csv(args[2])
        temp.to_csv(tmp_path / 'output.csv', index=False)

    def mock_check_file(*args):
        return False

    monkeypatch.setattr(
        'insurance_sell.extract.send_file_to_storage',
        mock_put_file_in_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.get_file_from_storage',
        mock_get_file_from_storage,
    )

    monkeypatch.setattr(
        'insurance_sell.extract.check_if_object_exists',
        mock_check_file,
    )

    extractor.extract(
        tmp_path / 'output.csv',
        client=client,
        overwrite=False,
        persist_in_storage=False,
    )
    # HACK: persist_in_local is submited, so read_csv will get data
    # before saved
    sleep(0.5)

    result = pd.read_csv(tmp_path / 'output.csv')

    assert len(result) == expected_len_data
