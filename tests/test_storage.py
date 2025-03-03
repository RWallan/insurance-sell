    check_if_object_exists,
    client,
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
