from pathlib import Path
from typing import Optional

import pandas as pd
from cyclopts import App
from rich.console import Console
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
    if not model:
        model_path = max(
            (Path().cwd() / 'model').glob('model_*.pkl'),
            key=lambda file: file.stat().st_ctime,
        )
    else:
        model_path = model

    loaded_model = get_model(model_path)

    console.print(loaded_model['metrics'])
