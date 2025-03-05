from typing import Optional

import pandas as pd
from feature_engine.dataframe_checks import check_X
from feature_engine.variable_handling import check_all_variables
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted


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
