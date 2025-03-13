from http import HTTPStatus
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insurance_sell.helpers.settings import Settings
from insurance_sell.helpers.utils import get_model

mlflow.set_tracking_uri(Settings().MLFLOW_TRACKING_URI)
loaded_model = get_model('models:/insurance-sell@production')

app = FastAPI()


@app.get('/healthcheck/')
def healthcheck():
    return {'status': 'ok'}


class RawInsuranceSell(BaseModel):
    Gender: str
    Age: int
    HasDrivingLicense: int
    Switch: int
    VehicleAge: str
    PastAccident: str
    AnnualPremium: str
    cohort: float


class PredictedValue(BaseModel):
    predicted: int


class Metric(BaseModel):
    metric: str
    value: float


class MetricList(BaseModel):
    metrics: list[Metric]


@app.post(
    '/predict/', status_code=HTTPStatus.CREATED, response_model=PredictedValue
)
def predict(input: RawInsuranceSell):
    data_ = pd.DataFrame(input.model_dump(exclude='cohort'), index=[0])

    result = loaded_model['model'].predict_proba(  # type: ignore
        data_[loaded_model['features']]
    )
    proba = result[:, 1]
    pred = (proba > input.cohort).astype(int)

    return {'predicted': int(pred)}


@app.get('/metrics/{metric}', response_model=MetricList)
def metrics(metric: Literal['train', 'test']):
    metrics = [
        Metric(metric=key.split('_')[1], value=round(item, 4))
        for key, item in loaded_model['metrics'].items()
        if metric in key
    ]

    return MetricList(metrics=metrics)
