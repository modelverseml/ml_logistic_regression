"""From-scratch binary logistic regression.

Trains a binary logistic regression with batch gradient descent on the binary
cross-entropy (log-loss) objective:

    p_i  = sigmoid(X_i beta)         # predicted probability of class 1
    L(beta) = -(1/n) sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

The gradient has the same clean form as linear regression:

    grad L = (1/n) X^T (p - y)

so the update rule is

    beta := beta - alpha * (1/n) X^T (sigmoid(X beta) - y)

Scale the inputs (e.g. with StandardScaler) for stable convergence, and encode
y as 0/1.
"""

import numpy as np
import pandas as pd


def _sigmoid(z):
    """Numerically stable sigmoid (avoids overflow for large negative z)."""
    # Split by sign so the exp arguments stay bounded in both branches.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegressionModel:
    def __init__(self, X_train, y_train, learning_rate=0.1, n_iterations=5000):
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.feature_names = list(X_train.columns)
        self.coefficients = None

    def build_model(self):
        """Fit with batch gradient descent on binary cross-entropy."""
        X = self.X_train.to_numpy(dtype=float)
        y = self.y_train.to_numpy(dtype=float)

        # Prepend a column of 1s so the first coefficient is the intercept.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.feature_names = ["Intercept"] + self.feature_names

        n_samples = X.shape[0]
        beta = np.zeros(X.shape[1])

        for _ in range(self.n_iterations):
            # Gradient of binary cross-entropy is (1/n) X^T (p - y), the same
            # shape as the OLS gradient, hence the same update form.
            probs = _sigmoid(X @ beta)
            gradient = X.T @ (probs - y) / n_samples
            beta = beta - self.learning_rate * gradient

        self.coefficients = beta

    def predict_proba(self, X):
        """Return P(y = 1 | X) for each row of X."""
        X = np.asarray(X, dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return _sigmoid(X @ self.coefficients)

    def predict(self, X, threshold=0.5):
        """Hard class predictions, using threshold on the predicted probability."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs."""
        return pd.DataFrame({
            "Feature": self.feature_names,
            "Coefficient": self.coefficients.round(3),
        })
