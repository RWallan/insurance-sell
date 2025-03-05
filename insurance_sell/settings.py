from pydantic_settings import BaseSettings, SettingsConfigDict


class MinioSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str


class Settings(BaseSettings):
    DATA_SOURCES: list[str] = [
        'https://raw.githubusercontent.com/prsdm/mlops-project/refs/heads/main/data/train.csv',
        'https://raw.githubusercontent.com/prsdm/mlops-project/refs/heads/main/data/test.csv',
    ]
    DATA_SOURCES_BUCKET: str = 'raw'
    OUTPUT_DATA_BUCKET: str = 'raw.csv'
