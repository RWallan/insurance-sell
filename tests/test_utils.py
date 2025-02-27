from datetime import datetime

import pandas as pd
from freezegun import freeze_time
from sklearn.pipeline import Pipeline

from insurance_sell.utils import get_model, save_model


def test_save_model(tmp_path):
    pipe = Pipeline(steps=[])

    model_file_path = save_model(tmp_path, pipe, {}, [])

    assert model_file_path in tmp_path.iterdir()


def test_get_model(tmp_path):
    with freeze_time('2025-01-01'):
        pipe = Pipeline(steps=[])
        expected = pd.Series(
            {
                'model': pipe,
                'features': [],
                'metrics': {},
                'date': datetime.now(),
            }
        )

        model_file_path = save_model(tmp_path, pipe, {}, [])

        model = get_model(model_file_path)

    pd.testing.assert_index_equal(model.index, expected.index)
