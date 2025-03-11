import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from cyclopts import App
from minio import Minio
from prefect import flow
from rich.console import Console
from rich.table import Table
from sklearn import model_selection

from insurance_sell import __version__
from insurance_sell.extract import Extractor
from insurance_sell.helpers.settings import (
    MinioSettings,
    ModelSettings,
    Settings,
)
from insurance_sell.helpers.storage import (
    get_file_from_storage,
)
from insurance_sell.helpers.utils import get_model, save_model
from insurance_sell.modeling.train import Trainer

console = Console()
client = Minio(
    MinioSettings().MINIO_ENDPOINT,  # type: ignore
    access_key=MinioSettings().MINIO_ACCESS_KEY,  # type: ignore
    secret_key=MinioSettings().MINIO_SECRET_KEY,  # type: ignore
    secure=False,
)
mlflow.set_tracking_uri('http://localhost:5000')

app = App(version=__version__)


@app.command
def extract(*, overwrite: bool = False):
    """Extract data from https://github.com/prsdm/mlops-project/tree/main and export to minIO storage.

    Args:
        overwrite: If True, the data extracted will overwrite existing data.
    """  # noqa: E501
    extractor = Extractor(Settings())

    extractor.extract(
        Settings().OUTPUT_DATA_BUCKET,
        client=client,
        overwrite=overwrite,
    )

    console.print(
        (
            f'File {Settings().OUTPUT_DATA_BUCKET} is being uploaded as object'
            f' to bucket {Settings().DATA_SOURCES_BUCKET}.'
        )
    )


@app.command
def fit(bucket_name: str, filename: str, random_state: int = 12):
    """Fit model.

    Args:
        bucket_name: Name of minIO bucket.
        filename: Filename of minIO object.
        random_state: Random seed to split data into train and test.
    """
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        df = get_file_from_storage(
            client, bucket_name, filename, f.name, to_df=True
        )
    if not isinstance(df, pd.DataFrame):
        console.print('Something went wrong while downloading data.')
        sys.exit(1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df[ModelSettings().features],  # type: ignore
        df[ModelSettings().target],  # type: ignore
        random_state=random_state,
        train_size=0.8,
        stratify=df[ModelSettings().target],  # type: ignore
    )

    with mlflow.start_run(run_name=f'run_{datetime.now()}') as run:
        trainer = Trainer(ModelSettings(), run.info.run_id)  # type: ignore
        mlflow.set_tag('developer', 'RWallan')
        trainer.fit(
            X_train,  # type: ignore
            y_train,  # type: ignore
            True,
            X_test=X_test,  # type: ignore
            y_test=y_test,  # type: ignore
        )

        saved_model = save_model('insurance-sell', run.info.run_id)

    console.print(f'Saved model: {saved_model}')


@app.command
@flow(log_prints=True)
def metrics(model: Optional[str] = None):
    if not model:
        model = 'models:/insurance-sell@production'
    loaded_model = get_model(model)

    console.print(loaded_model['metrics'])


@app.command
@flow(log_prints=True)
def predict(
    data: dict | str | Path,
    *,
    model: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    cohort: float = 0.5,
):
    if not model:
        model = 'models:/insurance-sell@production'
    if isinstance(data, dict):
        data_ = pd.DataFrame(data)
    else:
        # TODO: must force or adapt to file extension
        data_ = pd.read_csv(data)

    loaded_model = get_model(model)

    result = loaded_model['model'].predict_proba(  # type: ignore
        data_[loaded_model['features']]
    )
    proba = result[:, 1]
    pred = (proba > cohort).astype(int)

    data_ = data_.assign(proba=proba, pred=pred)

    if output_path:
        data_.to_csv(output_path, index=False)

    table = Table('id', 'proba', 'pred')
    for _, row in data_.head(5).iterrows():
        table.add_row(
            str(row['id']),
            str(round(row['proba'], 2)),  # type: ignore
            str(row['pred']),
        )

    console.print(table)
