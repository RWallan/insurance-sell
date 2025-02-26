import logging
from typing import Optional

import pandas as pd
from feature_engine import encoding, imputation
from feature_engine.dataframe_checks import check_X
from feature_engine.variable_handling import check_all_variables
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.pipeline import Pipeline

FEATURES = [
    'Gender',
    'Age',
    'HasDrivingLicense',
    'Switch',
    'VehicleAge',
    'PastAccident',
    'AnnualPremium',
]
TARGET = 'Result'

logger = logging.getLogger(__name__)


class StringCleaner(BaseEstimator, TransformerMixin):
    """The StringCleaner() replaces the pattern with a target and transform
    the variable to a float type.
    """  # noqa: D205

    def __init__(
        self,
        variables: int | str | list[int | str],
        pattern_replace: dict[str, str],
    ) -> None:
        """Init the variables and patterns.

        Args:
            variables: Variables in DataFrame to be transformed.
            pattern_replace: A dict with {pattern: replace}.
        """
        super().__init__()
        self.variables = variables
        self.pattern_replace = pattern_replace

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # noqa: N803
        """Fit the variables to dataframe.

        Args:
            X: DataFrame.
            y: Target. It is not needed in this encoded.
        """
        # Check that input is a dataframe
        X = check_X(X)

        # Check if all variables listed is in dataframe
        variables_ = check_all_variables(X, self.variables)

        self.n_features_in = len(X.columns)
        self.variables_ = variables_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Replace patterns and transform the columns to float.

        Args:
            X: DataFrame.

        Returns:
            Same DataFrame with columns tranformed.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # Check that input is a dataframe
        X = check_X(X)

        # Check if the dataframe has the same number as used in fit method
        if len(X.columns) != self.n_features_in:
            raise ValueError(
                'The number of columns in this dataset is different from the '
                'one used to fit this transformer.'
            )

        for pattern, replace in self.pattern_replace.items():
            X[self.variables_] = X[self.variables_].apply(
                lambda col: col.str.replace(pattern, replace)
            )
        X[self.variables_] = X[self.variables_].astype(float)

        return X


# TODO: Use some settings to create this pipeline.
# TODO: Create more unit tests
def create_pipeline() -> Pipeline:
    """Create transform Pipeline."""
    logger.info(
        (
            'StringCleaner: Using `AnnualPremium` with these patterns: '
            "{'£': '', ',': ''}"
        )
    )
    cleaner = StringCleaner(
        variables='AnnualPremium', pattern_replace={'£': '', ',': ''}
    )

    logger.info(
        (
            'CategoricalImputer: Imputing `Gender` and `VehicleAge` with '
            'frequent imputation method'
        )
    )
    frequent_imputer = imputation.CategoricalImputer(
        variables=['Gender', 'VehicleAge'],
        imputation_method='frequent',
    )

    logger.info('MeanMedianImputer: Imputing `Age` with median')
    median_imputer = imputation.MeanMedianImputer(
        variables='Age',
        imputation_method='median',
    )

    logger.info(
        (
            'CategoricalImputer: Imputing `PastAccident` with missing method.'
            'Imputing `Unknown`'
        )
    )
    missing_imputer = imputation.CategoricalImputer(
        variables='PastAccident',
        imputation_method='missing',
        fill_value='Unknown',
    )

    logger.info('OneHotEncoder: Tranforming `PastAccident` dropping last')
    one_hot = encoding.OneHotEncoder(variables='PastAccident', drop_last=True)

    logger.info('ArbitraryNumberImputer: Imputing `1` to `HasDrivingLicense`')
    arbitrary_positive_one = imputation.ArbitraryNumberImputer(
        variables='HasDrivingLicense', arbitrary_number=1
    )

    logger.info('ArbitraryNumberImputer: Imputing `-1` to `Switch`')
    arbitrary_negative_one = imputation.ArbitraryNumberImputer(
        variables='Switch', arbitrary_number=-1
    )

    return Pipeline(
        steps=[
            ('StringCleaner', cleaner),
            ('FrequentImputer', frequent_imputer),
            ('MedianImputer', median_imputer),
            ('MissingImputer', missing_imputer),
            ('OneHot', one_hot),
            ('ArbitraryPositiveOne', arbitrary_positive_one),
            ('ArbitraryNegativeOne', arbitrary_negative_one),
        ]
    )
