# ruff: noqa: D102
from typing import Iterator, Optional, Protocol, TypedDict

import pandas as pd
from minio import S3Error
from minio.commonconfig import ENABLED
from minio.versioningconfig import VersioningConfig
from prefect import task
from prefect.cache_policies import NONE
from prefect.logging import get_run_logger


class BucketClient(Protocol):
    def bucket_exists(self, bucket_name: str): ...
    def fget_object(
        self, bucket_name: str, object_name: str, file_path: str
    ): ...
    def fput_object(
        self, bucket_name: str, object_name: str, file_path: str
    ): ...
    def list_objects(self, bucket_name: str) -> Iterator: ...
    def make_bucket(self, bucket_name: str): ...
    def set_bucket_versioning(
        self, bucket_name: str, config: VersioningConfig
    ): ...


def _create_bucket_if_not_exists(
    client: BucketClient, bucket_name: str, logger
):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        client.set_bucket_versioning(bucket_name, VersioningConfig(ENABLED))
    else:
        logger.info(f'Bucket {bucket_name} already exists. Ignoring...')


class ObjectInfo(TypedDict):
    output_file: str
    bucket_name: str


@task(cache_policy=NONE)
def send_file_to_storage(
    client: BucketClient, bucket_name, input_file, output_file
):
    logger = get_run_logger()
    _create_bucket_if_not_exists(client, bucket_name, logger)

    try:
        client.fput_object(bucket_name, output_file, input_file)
        logger.info(
            (
                f'{input_file} successfully uploaded as object {output_file} '
                f'to bucket {bucket_name}.'
            )
        )
    except S3Error as e:
        logger.error(f'Error occurred while uploading file to storage: {e}')


@task
def check_if_object_exists(client: BucketClient, bucket_name, object_name):
    logger = get_run_logger()
    try:
        objects = client.list_objects(bucket_name)
        filtered_objects = filter(
            lambda obj: obj.object_name == object_name, objects
        )
        return True if any(filtered_objects) else False
    except S3Error:
        logger.error(f'Bucket {object_name} not found')
        return False


@task(cache_policy=NONE)
def get_file_from_storage(
    client, bucket_name, file, output_file, to_df: bool = True
) -> Optional[pd.DataFrame]:
    obj = client.fget_object(bucket_name, file, output_file)

    if to_df and obj.object_name:
        return pd.read_csv(output_file)
