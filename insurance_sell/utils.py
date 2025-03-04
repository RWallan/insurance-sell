import logging
from typing import TypedDict

import mlflow
import mlflow.models
import mlflow.sklearn
from imblearn.pipeline import Pipeline
from mlflow.models.model import ModelInfo

logger = logging.getLogger(__name__)


class LoadedModel(TypedDict):
    features: list[str]
    model: Pipeline
    metrics: dict


def save_model(model_name, run_id):
    info = mlflow.register_model(f'runs:/{run_id}/model', model_name)
    logger.info(
        f"""Model {info.name} succesfully registered as {info.status}.
Model version: {info.version}"""
    )

    return info.run_id


def _get_metrics(run_id):
    return mlflow.get_run(run_id).data.metrics


def _get_features(model_info: ModelInfo):
    return [i['name'] for i in model_info.signature.inputs.to_dict()]


def get_model(uri) -> LoadedModel:
    model = mlflow.sklearn.load_model(uri)
    model_info = mlflow.models.get_model_info(uri)
    if not model:
        logger.error(f'Model {uri} not found')
        raise Exception(f'Model {uri} not found')
    features = _get_features(model_info)
    metrics = _get_metrics(model_info.run_id)

    return {'features': features, 'model': model, 'metrics': metrics}
