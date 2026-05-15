"""
Automated Backward Feature Selection (Logistic Regression)
----------------------------------------------------------
Iteratively removes one feature at a time until every remaining feature is
both statistically significant (p-value <= 0.05) and not collinear with the
rest (VIF <= 10). At each step the single "worst" feature is dropped:

    - Both checks fail: drop the highest-VIF feature (ties broken by p-value).
    - Only p-values too high: drop the largest p-value.
    - Only VIFs too high:      drop the largest VIF.

The p-values come from a statsmodels GLM with a Binomial family — the
maximum-likelihood logistic regression — so the same Wald test logic from
linear OLS applies to the log-odds coefficients here.
"""

import pandas as pd
import statsmodels.api as sm

import variance_inflation_factor_data as vif_module


def final_data(X_train, y_train):
    """Return the list of surviving features after backward elimination."""

    features = list(X_train.columns)

    while True:
        p_values = get_p_values(X_train[features], y_train)
        vif_values = get_vif_values(X_train[features])

        features_data = pd.merge(p_values, vif_values, on='Features', how='inner')

        high_vif = (features_data['VIF'] > 10).any()
        high_p = (features_data['p-value'] > 0.05).any()

        # Stop once no feature violates either threshold.
        if not (high_vif or high_p):
            print(features_data)
            return features

        if high_vif and high_p:
            # Both bad: prefer dropping the most collinear; break ties by p-value.
            worst_feature = features_data.sort_values(
                ['VIF', 'p-value'], ascending=False
            ).iloc[0]["Features"]
        elif high_p:
            worst_feature = features_data.sort_values(
                'p-value', ascending=False
            ).iloc[0]["Features"]
        else:  # high_vif only
            worst_feature = features_data.sort_values(
                'VIF', ascending=False
            ).iloc[0]["Features"]

        features.remove(worst_feature)


def get_vif_values(X_train):
    """Compute VIF for each feature; see variance_inflation_factor_data.VIF."""

    return vif_module.VIF(X_train).get_vif_values()


def get_p_values(X_train, y_train):
    """Fit a Binomial GLM (logistic regression) and return per-feature p-values."""

    X_train_sm = sm.add_constant(X_train)

    # GLM with Binomial family == logistic regression via MLE.
    sm_lr = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial()).fit()

    p_values = pd.DataFrame({'Features': X_train.columns})
    # GLM returns p-values for [const, feat_1, ...]; drop the const entry.
    p_values['p-value'] = list(round(sm_lr.pvalues, 4))[1:]

    return p_values
