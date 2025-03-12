import numpy as np
import pandas as pd
from prefect.logging.loggers import disable_run_logger

from insurance_sell.modeling.train import Trainer
from insurance_sell.modeling.transformers import StringCleaner


def test_init_pipelines(test_model_settings, client):
    trainer = Trainer(test_model_settings, '1234', client)
    with disable_run_logger():
        trainer.create_pipeline.fn(trainer)

    assert len(trainer._transformers) == 1
    assert trainer._transformers[0][0] == 'StringCleaner'
    assert isinstance(trainer._transformers[0][1], StringCleaner)


def test_init_model(test_model_settings, client):
    trainer = Trainer(test_model_settings, '1234', client)
    with disable_run_logger():
        trainer.configure_model.fn(trainer)

    assert isinstance(
        trainer._model, test_model_settings.model_selection.model
    )


def test_report_metrics(test_model_settings, client):
    trainer = Trainer(test_model_settings, '1234', client)
    y_true = pd.Series([1, 0, 0, 1, 1, 0])
    y_proba = np.array(
        [[0.3, 0.7], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8], [0, 1], [1, 0]]
    )
    metrics = trainer.report_metrics(y_true, y_proba, cohort=0.5)
    assert metrics == {
        'Accuracy': 1.0,
        'ROC AUC': np.float64(1.0),
        'Precision': 1.0,
        'Recall': 1.0,
    }
