import pandas as pd

from insurance_sell.settings import MinioSettings
from insurance_sell.storage import (
    check_if_object_exists,
    client,
    get_file_from_storage_as_df,
    send_file_to_storage,
)

settings = MinioSettings()  # type: ignore


def test_send_file_to_storage(tmp_path):
    file = tmp_path / 'test.csv'
    file.touch(exist_ok=True)
    send_file_to_storage(file, 'test.csv')

    obj = client.stat_object(settings.BUCKET_NAME, 'test.csv')

    assert obj.object_name == 'test.csv'

    client.remove_object(settings.BUCKET_NAME, 'test.csv')


def test_check_if_object_exists_must_return_true_if_exists(tmp_path):
    file = tmp_path / 'test.csv'
    file.touch(exist_ok=True)
    send_file_to_storage(file, 'test.csv')

    assert check_if_object_exists(settings.BUCKET_NAME, 'test.csv')

    client.remove_object(settings.BUCKET_NAME, 'test.csv')


def test_check_if_object_exists_must_return_false_if_not_exists():
    assert not check_if_object_exists(settings.BUCKET_NAME, 'wrong.csv')


def test_get_file_from_object(tmp_path):
    file = tmp_path / 'test.csv'
    test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    test_data.to_csv(file, index=False)
    send_file_to_storage(file, 'test.csv')

    df = get_file_from_storage_as_df(
        settings.BUCKET_NAME, 'test.csv', tmp_path / 'received.csv'
    )

    assert isinstance(df, pd.DataFrame)

    client.remove_object(settings.BUCKET_NAME, 'test.csv')
