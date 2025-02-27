import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from insurance_sell.pipeline import (
    StringCleaner,
    create_pipeline,
    report_metrics,
)


def test_string_cleaner():
    sample_df = pd.DataFrame({'column': ['R$ 2,000.00', 'R$ 1,500.00']})

    cleaner = StringCleaner('column', {'R$': '', ',': ''})

    cleaner.fit(sample_df)

    result = cleaner.transform(sample_df)

    pd.testing.assert_frame_equal(
        result, pd.DataFrame({'column': [2000.0, 1500.0]})
    )


def test_string_cleaner_must_raise_if_not_fitted():
    sample_df = pd.DataFrame({'column': ['R$ 2,000.00', 'R$ 1,500.00']})

    cleaner = StringCleaner('column', {'R$': '', ',': ''})

    with pytest.raises(NotFittedError):
        cleaner.transform(sample_df)


def test_steps_created_in_pipeline():
    expected_len = 7
    pipeline = create_pipeline()

    assert len(pipeline.steps) == expected_len


def test_report_metrics():
    y_true = pd.Series([1, 0, 0, 1, 1, 0])
    y_proba = np.array(
        [[0.3, 0.7], [0.6, 0.4], [0.8, 0.2], [0.2, 0.8], [0, 1], [1, 0]]
    )
    metrics = report_metrics(y_true, y_proba, cohort=0.5)
    assert metrics == {
        'Accuracy': 1.0,
        'ROC AUC': np.float64(1.0),
        'Precision': 1.0,
        'Recall': 1.0,
    }
