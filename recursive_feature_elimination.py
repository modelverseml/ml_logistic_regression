"""
Recursive Feature Elimination (RFE) for Logistic Regression
-----------------------------------------------------------
Thin wrapper around scikit-learn's RFE backed by `LogisticRegression`. RFE
fits the estimator, ranks features by their absolute coefficient, drops the
weakest, and repeats until only `n_features_to_select` remain. Useful when
you want a fixed top-k subset for downstream modelling.

Note: For logistic regression, scale your features first (e.g. with
`StandardScaler`) — RFE's ranking is based on coefficient magnitude, which
is only comparable when features are on the same scale.
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


class RfeClass:

    def __init__(self, X_train, y_train, number_of_features):

        self.X_train = X_train
        self.y_train = y_train
        self.number_of_features = number_of_features
        self.lm = None
        self.rfe = None
        self.top_columns = None

    def get_rfe_output(self):
        """Fit RFE and return the names of the selected columns."""

        self.lm = LogisticRegression(max_iter=1000).fit(self.X_train, self.y_train)
        self.rfe = RFE(self.lm, n_features_to_select=self.number_of_features)
        self.rfe.fit(self.X_train, self.y_train)

        # rfe.support_ is a boolean mask aligned with X_train.columns.
        self.top_columns = self.X_train.columns[self.rfe.support_]
        return self.top_columns
