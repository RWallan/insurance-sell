from insurance_sell.extract import extract_data


def test_extract_data(tmp_path):
    extract_data(tmp_path / 'raw.csv')

    assert (tmp_path / 'raw.csv').exists()
