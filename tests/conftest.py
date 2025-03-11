from pathlib import Path

import pytest
from minio import Minio
from prefect.testing.utilities import prefect_test_harness
from pydantic_settings import SettingsConfigDict

from insurance_sell.settings import MinioSettings, ModelSettings


@pytest.fixture
def client():
    settings = MinioSettings()  # type: ignore
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False,
    )


@pytest.fixture(autouse=True, scope='session')
def prefect_test_fixture():
    with prefect_test_harness():
        yield


@pytest.fixture
def test_model_settings(monkeypatch):
    TEST_CONFIG_FILE = str(Path().cwd() / 'tests' / 'test_model_config.toml')
    monkeypatch.setattr(
        'insurance_sell.settings.ModelSettings.model_config',
        SettingsConfigDict(toml_file=TEST_CONFIG_FILE),
    )
    return ModelSettings()  # type: ignore
