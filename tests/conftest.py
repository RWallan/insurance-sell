import pytest
from minio import Minio
from prefect.testing.utilities import prefect_test_harness

from insurance_sell.settings import MinioSettings


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
