import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from insurance_sell.pipeline import StringCleaner


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
