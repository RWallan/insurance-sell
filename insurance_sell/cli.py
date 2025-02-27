from pathlib import Path
from typing import Optional

import pandas as pd
from cyclopts import App
from rich.console import Console
from rich.table import Table
from sklearn import model_selection

from insurance_sell import __version__, pipeline
from insurance_sell.extract import extract_data
from insurance_sell.utils import get_model, save_model

console = Console()
app = App(version=__version__)


@app.command
def extract(overwrite: bool = False):
    """Extract data from https://github.com/prsdm/mlops-project/tree/main and export to csv.

    Args:
        overwrite: If True, the data extracted will overwrite existing data.
    """  # noqa: E501
    file = extract_data(overwrite)
    console.print(f'Raw data created at: {file}.')


@app.command
def train(file_path: str | Path = Path().cwd() / 'data' / 'raw.csv'):
    output_path = Path().cwd() / 'model'
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df[pipeline.FEATURES],
        df[pipeline.TARGET],
        random_state=12,
        train_size=0.8,
        stratify=df[pipeline.TARGET],
    )

    model = pipeline.fit_model(X_train, y_train)  # type: ignore
    metrics = pipeline.evalute_model(X_train, y_train, X_test, y_test, model)  # type: ignore

    saved_model = save_model(output_path, model, metrics, pipeline.FEATURES)

    console.print(f'Saved model at: {saved_model}')


@app.command
def metrics(model: Optional[str] = None):
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
