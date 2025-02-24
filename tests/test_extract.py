from insurance_sell.extract import extract_data


def test_extract_data(tmp_path, monkeypatch):
    monkeypatch.setattr('insurance_sell.extract.DATA_PATH', tmp_path)

    extract_data()

    assert (tmp_path / 'raw.csv').exists()
