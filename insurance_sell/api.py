from http import HTTPStatus

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insurance_sell.helpers.utils import get_model

mlflow.set_tracking_uri('http://localhost:5000')
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


@app.get('/metrics/')
def metrics():
    return loaded_model['metrics']
