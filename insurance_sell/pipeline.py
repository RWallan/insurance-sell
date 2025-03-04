from typing import Optional

import mlflow
import mlflow.models
import mlflow.sklearn
import pandas as pd
from feature_engine import encoding, imputation
from feature_engine.dataframe_checks import check_X
from feature_engine.variable_handling import check_all_variables
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from prefect import task
from prefect.cache_policies import NONE
from prefect.logging import get_run_logger
from sklearn import ensemble, metrics, model_selection
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

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
@task
def create_pipeline() -> list[tuple]:
    """Create transform Pipeline."""
    logger = get_run_logger()
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

    logger.info(
        (
            'OneHotEncoder: Transforming '
            '`PastAccident`, `Gender` and `VehicleAge` dropping last'
        )
    )
    one_hot = encoding.OneHotEncoder(
        variables=['PastAccident', 'Gender', 'VehicleAge'], drop_last=True
    )

    logger.info('ArbitraryNumberImputer: Imputing `1` to `HasDrivingLicense`')
    arbitrary_positive_one = imputation.ArbitraryNumberImputer(
        variables='HasDrivingLicense', arbitrary_number=1
    )

    logger.info('ArbitraryNumberImputer: Imputing `-1` to `Switch`')
    arbitrary_negative_one = imputation.ArbitraryNumberImputer(
        variables='Switch', arbitrary_number=-1
    )

    mlflow.set_tag(
        'preprocessing',
        (
            'StringCleaner, FrequentImputer, MedianImputer, MissingImputer, '
            'OneHot, ArbitraryPositiveOne, ArbitraryNegativeOne',
        ),
    )

    return [
        ('StringCleaner', cleaner),
        ('FrequentImputer', frequent_imputer),
        ('MedianImputer', median_imputer),
        ('MissingImputer', missing_imputer),
        ('OneHot', one_hot),
        ('ArbitraryPositiveOne', arbitrary_positive_one),
        ('ArbitraryNegativeOne', arbitrary_negative_one),
    ]


# TODO: Create a external settings
@task
def configure_model() -> model_selection.GridSearchCV:
    """Create a GridSearchCV with multiple model parameters."""
    logger = get_run_logger()
    # TODO: Create a external setting to set random_state globally
    model = ensemble.RandomForestClassifier(random_state=12)

    parameters = {
        'min_samples_leaf': [10, 25],
        'n_estimators': [100],
        'criterion': ['gini'],
        'max_depth': [5, 10],
    }
    logger.info(f'GridSearchCV parameters: {parameters}')

    return model_selection.GridSearchCV(
        model, param_grid=parameters, cv=3, n_jobs=-1, verbose=3
    )


def get_choosed_params(model: Pipeline):
    return model.steps[-1][1].best_params_


@task(cache_policy=NONE)
def fit_model(X: pd.DataFrame, y: pd.Series, run_id):  # noqa: N803
    # Resampling before pipeline because some incompatibilities is happening
    # and I dunno how to fix it
    logger = get_run_logger()
    pipelines = create_pipeline()
    model = configure_model()

    mlflow.set_tag('resampling', 'SMOTE')
    pipelines.extend(
        [
            ('smote', SMOTE(sampling_strategy=1.0, random_state=12)),  # type: ignore
            ('model', model),
        ]
    )

    model_pipeline = Pipeline(steps=pipelines)
    model_pipeline.fit(X, y)
    parameters = get_choosed_params(model_pipeline)
    logger.info(f'Choosed params: {parameters}')
    mlflow.log_params(parameters, run_id=run_id)
    signature = mlflow.models.infer_signature(model_input=X, params=parameters)
    mlflow.sklearn.log_model(model_pipeline, 'model', signature=signature)

    return model_pipeline


@task
def report_metrics(y_true, y_proba, cohort: float, prefix: str = ''):
    y_pred = (y_proba[:, 1] > cohort).astype(int)

    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)

    return {
        f'{prefix}Accuracy': acc,
        f'{prefix}ROC AUC': auc,
        f'{prefix}Precision': precision,
        f'{prefix}Recall': recall,
    }


# TODO: Learn how to test it
@task
def evalute_model(  # noqa: PLR0913
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    X_test: pd.DataFrame,  # noqa: N803
    y_test: pd.Series,
    model: Pipeline,
    run_id,
    cohort: float = 0.5,
):
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    train_metrics = report_metrics(
        y_train, y_train_proba, cohort, prefix='train_'
    )
    mlflow.log_metrics(train_metrics, run_id=run_id)
    test_metrics = report_metrics(y_test, y_test_proba, cohort, prefix='test_')
    mlflow.log_metrics(test_metrics, run_id=run_id)

    return {'Train Metrics': train_metrics, 'Test Metrics': test_metrics}
