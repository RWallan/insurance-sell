import pandas as pd
from prefect.logging.loggers import disable_run_logger

from insurance_sell.settings import MinioSettings
from insurance_sell.storage import (
    check_if_object_exists,
    get_file_from_storage,
    send_file_to_storage,
)

settings = MinioSettings()  # type: ignore


def test_send_file_to_storage(client, tmp_path):
    file = tmp_path / 'test.csv'
    file.touch(exist_ok=True)
    with disable_run_logger():
        send_file_to_storage.fn(client, 'raw', file, 'test.csv')

    obj = client.stat_object('raw', 'test.csv')

    assert obj.object_name == 'test.csv'

    client.remove_object('raw', 'test.csv')


def test_check_if_object_exists_must_return_true_if_exists(client, tmp_path):
    file = tmp_path / 'test.csv'
    file.touch(exist_ok=True)
    with disable_run_logger():
        send_file_to_storage.fn(client, 'raw', file, 'test.csv')

        assert check_if_object_exists.fn(client, 'raw', 'test.csv')

    client.remove_object('raw', 'test.csv')


def test_check_if_object_exists_must_return_false_if_not_exists(client):
    with disable_run_logger():
        assert not check_if_object_exists.fn(client, 'raw', 'wrong.csv')


def test_check_if_object_exists_must_return_false_if_wrong_bucket(client):
    with disable_run_logger():
        assert not check_if_object_exists.fn(client, 'wrong', 'wrong.csv')


def test_get_file_from_object(tmp_path, client):
    file = tmp_path / 'test.csv'
    test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    test_data.to_csv(file, index=False)
    with disable_run_logger():
        send_file_to_storage.fn(client, 'raw', file, 'test.csv')

        df = get_file_from_storage(
            client, 'raw', 'test.csv', tmp_path / 'received.csv'
        )

    assert isinstance(df, pd.DataFrame)

    client.remove_object('raw', 'test.csv')
