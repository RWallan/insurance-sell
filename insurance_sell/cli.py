import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from cyclopts import App
from rich.console import Console
from rich.table import Table
from sklearn import model_selection

from insurance_sell import __version__, pipeline
from insurance_sell.extract import extract_data
from insurance_sell.storage import (
    check_if_object_exists,
    get_file_from_storage,
    send_file_to_storage,
)
from insurance_sell.utils import get_model, save_model

console = Console()
app = App(version=__version__)

mlflow.set_tracking_uri('http://localhost:5000')


@app.command
def extract(overwrite: bool = False):
    """Extract data from https://github.com/prsdm/mlops-project/tree/main and export to minIO storage.

    Args:
        overwrite: If True, the data extracted will overwrite existing data.
    """  # noqa: E501
    has_object = False
    if not overwrite:
        has_object = check_if_object_exists('raw', 'raw.csv')
    overwrite_ = False if has_object and not overwrite else True

    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        if not overwrite_:
            get_file_from_storage('raw', 'raw.csv', f.name)
        extract_data(f.name, overwrite_)
        info = send_file_to_storage(f.name, 'raw.csv')

    console.print(
        (
            f'{info["output_file"]} successfully uploaded as object '
            f'to bucket {info["bucket_name"]}.'
        )
    )


@app.command
def train(bucket_name: str, filename: str):
    """Fit model.

    Args:
        bucket_name: Name of minIO bucket.
        filename: Filename of minIO object.
    """
    output_path = Path().cwd() / 'model'
    output_path.mkdir(exist_ok=True, parents=True)

    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        df = get_file_from_storage(bucket_name, filename, f.name, to_df=True)
    if not isinstance(df, pd.DataFrame):
        console.print('Something went wrong while downloading data.')
        sys.exit(1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df[pipeline.FEATURES],
        df[pipeline.TARGET],
        random_state=12,
        train_size=0.8,
        stratify=df[pipeline.TARGET],
    )

    with mlflow.start_run(run_name=f'run_{datetime.now()}') as run:
        mlflow.set_tag('developer', 'RWallan')
        model = pipeline.fit_model(X_train, y_train, run.info.run_id)  # type: ignore
        pipeline.evalute_model(
            X_train,  # type: ignore
            y_train,  # type: ignore
            X_test,  # type: ignore
            y_test,  # type: ignore
            model,
            run.info.run_id,
        )

        saved_model = save_model('insurance-sell', run.info.run_id)

    console.print(f'Saved model: {saved_model}')


@app.command
def metrics(model: Optional[str] = None):
    if not model:
        model = 'models:/insurance-sell@production'
    loaded_model = get_model(model)

    console.print(loaded_model['metrics'])


@app.command
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
