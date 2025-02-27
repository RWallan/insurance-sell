import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def save_model(
    path: str | Path, model: Pipeline, metrics: dict, features: str | list[str]
) -> Path:
    now = datetime.now()
    path_ = Path(path)

    model_file_path = path_ / f'model_{now}.pkl'

    obj = pd.Series(
        {'model': model, 'features': features, 'metrics': metrics, 'date': now}
    )

    obj.to_pickle(model_file_path)
    logger.info(f'Saved model to {str(model_file_path)}')

    return model_file_path


def get_model(path: Optional[str | Path] = None):
    if not path:
        path = max(
            (Path().cwd() / 'model').glob('model_*.pkl'),
            key=lambda file: file.stat().st_ctime,
        )
    return pd.read_pickle(path)
