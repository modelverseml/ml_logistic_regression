"""
statsmodels Logistic Regression (Binomial GLM) Wrapper
------------------------------------------------------
Fits the same logistic regression as scikit-learn but via statsmodels' GLM
with a Binomial family — useful because statsmodels also reports standard
errors, Wald z-statistics, and p-values for each coefficient (the basis for
the hypothesis-testing-driven feature selection in
`automated_feature_selection.py`).
"""

import pandas as pd
import statsmodels.api as sm


class SMLogisticModel:

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.lr = None

    def build_model(self):
        """Fit a Binomial GLM (i.e. logistic regression by MLE)."""

        # statsmodels does not add an intercept by default — add_constant prepends 1s.
        X_train_sm = sm.add_constant(self.X_train)
        self.lr = sm.GLM(
            self.y_train, X_train_sm, family=sm.families.Binomial()
        ).fit()

    def predict_proba(self, X):
        """Predicted probability P(y = 1 | X)."""

        return self.lr.predict(sm.add_constant(X))

    def predict(self, X, threshold=0.5):
        """Hard class predictions using `threshold` on the predicted probability."""

        return (self.predict_proba(X) >= threshold).astype(int)

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs."""

        return pd.DataFrame({
            "Feature": self.lr.params.index,
            "Coefficient": self.lr.params.values.round(3),
        })

    def summary(self):
        """Full GLM summary table with std-errors, z/p-values, deviance, AIC."""

        return self.lr.summary()
