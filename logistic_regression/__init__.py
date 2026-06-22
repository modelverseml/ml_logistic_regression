"""Logistic regression toolkit.

Three interchangeable ways to fit a binary logistic regression (a from-scratch
gradient-descent model plus scikit-learn and statsmodels wrappers), together
with the feature-selection and classification-metric tools used around them.
Every model exposes the same interface: build_model(), predict(X),
predict_proba(X) and get_parameters().

See README.md for the theory and examples/walkthrough.py for a runnable demo.
"""

from .automated_feature_selection import final_data
from .classification_metrics import ClassificationMetrics
from .logistic_regression_model_building import LogisticRegressionModel
from .recursive_feature_elimination import RfeClass
from .sklearn_logistic_model_building import SkLearnLogisticModel
from .statsmodel_logistic_model_building import SMLogisticModel
from .variance_inflation_factor_data import VIF

__all__ = [
    "LogisticRegressionModel",
    "SkLearnLogisticModel",
    "SMLogisticModel",
    "ClassificationMetrics",
    "RfeClass",
    "VIF",
    "final_data",
]

__version__ = "1.0.0"
