import pytest
from minio import Minio

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
