from pydantic_settings import BaseSettings, SettingsConfigDict


class MinioSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    BUCKET_NAME: str = 'raw'
