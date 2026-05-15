"""
scikit-learn Logistic Regression Wrapper
----------------------------------------
Thin wrapper around `sklearn.linear_model.LogisticRegression` that keeps the
same `build_model / predict / predict_proba / get_parameters` interface as the
from-scratch implementation. Forwards extra kwargs (e.g. `C`, `penalty`,
`class_weight='balanced'`) directly to scikit-learn.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


class SkLearnLogisticModel:

    def __init__(self, X_train, y_train, **logreg_kwargs):

        self.X_train = X_train
        self.y_train = y_train
        # `max_iter=1000` keeps the LBFGS solver from warning on real datasets.
        self.logreg_kwargs = {'max_iter': 1000, **logreg_kwargs}
        self.lr = None

    def build_model(self):

        self.lr = LogisticRegression(**self.logreg_kwargs).fit(
            self.X_train, self.y_train
        )

    def predict(self, X):

        return self.lr.predict(X)

    def predict_proba(self, X):
        """Return P(y = 1 | X) — second column of sklearn's `predict_proba`."""

        return self.lr.predict_proba(X)[:, 1]

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs, intercept last."""

        # sklearn stores coefficients as (1, n_features) for binary classification.
        coef_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Coefficient": self.lr.coef_[0].round(3),
        })
        coef_df.loc[len(coef_df)] = ["Intercept", round(float(self.lr.intercept_[0]), 3)]
        return coef_df
