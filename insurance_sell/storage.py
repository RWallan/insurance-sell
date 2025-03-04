from typing import Optional, TypedDict

import pandas as pd
from minio import Minio, S3Error
from minio.commonconfig import ENABLED
from minio.versioningconfig import VersioningConfig
from prefect import task
from prefect.logging import get_run_logger

from insurance_sell.settings import MinioSettings

settings = MinioSettings()  # type: ignore

client = Minio(
    'localhost:9000',
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False,
)


def _create_bucket_if_not_exists(bucket_name: str, logger):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        client.set_bucket_versioning(bucket_name, VersioningConfig(ENABLED))
    else:
        logger.info(f'Bucket {bucket_name} already exists. Ignoring...')


class ObjectInfo(TypedDict):
    output_file: str
    bucket_name: str


@task
def send_file_to_storage(input_file, output_file) -> ObjectInfo:
    logger = get_run_logger()
    _create_bucket_if_not_exists(settings.BUCKET_NAME, logger)

    try:
        client.fput_object(settings.BUCKET_NAME, output_file, input_file)
        logger.info(
            (
                f'{input_file} successfully uploaded as object {output_file} '
                f'to bucket {settings.BUCKET_NAME}.'
            )
        )
    except S3Error as e:
        logger.error(f'Error occurred while uploading file to storage: {e}')

    return {'output_file': output_file, 'bucket_name': settings.BUCKET_NAME}


@task
def check_if_object_exists(bucket_name, object_name):
    logger = get_run_logger()
    try:
        objects = client.list_objects(bucket_name)
        filtered_objects = filter(
            lambda obj: obj.object_name == object_name, objects
        )
        return True if any(filtered_objects) else False
    except S3Error:
        logger.error(f'Bucket {bucket_name} not found')
        return False


@task
def get_file_from_storage(
    bucket_name, file, output_file, to_df: bool = True
) -> Optional[pd.DataFrame]:
    obj = client.fget_object(bucket_name, file, output_file)

    if to_df and obj.object_name:
        return pd.read_csv(output_file)
