"""
Variance Inflation Factor (VIF)
-------------------------------
VIF quantifies how much the variance of a regression coefficient is inflated
because of linear dependence with other features. For feature X_j,

    VIF_j = 1 / (1 - R_j^2)

where R_j^2 is the R-squared of the regression of X_j on all other features.

Rules of thumb:
    VIF ≈ 1   -> no correlation with other features
    VIF 1–5  -> moderate, usually acceptable
    VIF > 10 -> problematic multicollinearity, consider dropping the feature

VIF applies to the design matrix and is independent of the link function,
so it is just as valid for logistic regression as for OLS.

Usage:
    vif_df = VIF(X_train).get_vif_values()
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VIF:

    def __init__(self, X_train=None):

        self.X_train = X_train if X_train is not None else pd.DataFrame()
        self.vif = pd.DataFrame(columns=['Features', 'VIF'])

    def get_vif_values(self):
        """Return a DataFrame of (feature, VIF) pairs sorted by VIF desc."""

        self.vif['Features'] = self.X_train.columns
        self.vif['VIF'] = [
            variance_inflation_factor(self.X_train.values, i)
            for i in range(self.X_train.shape[1])
        ]
        self.vif['VIF'] = round(self.vif['VIF'], 2)
        self.vif = self.vif.sort_values(by=['VIF'], ascending=False)

        return self.vif
