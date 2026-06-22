"""End-to-end logistic regression walkthrough.

Reproduces the old notebook as a single runnable script:

    1. build an imbalanced synthetic classification dataset
    2. train/test split (stratified) + feature scaling
    3. feature selection (RFE, then VIF + p-value backward elimination)
    4. fit three models (sklearn, statsmodels GLM, from-scratch GD)
    5. compare coefficients, then evaluate with the full metric suite
    6. sweep the decision threshold and try class_weight='balanced'
    7. (optional) show the diagnostic plots

Run from the repository root:

    python examples/walkthrough.py            # print the full walkthrough
    python examples/walkthrough.py --plot     # also show the diagnostic plots
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

from logistic_regression import (  # noqa: E402  (import after the sys.path tweak)
    ClassificationMetrics,
    LogisticRegressionModel,
    RfeClass,
    SkLearnLogisticModel,
    SMLogisticModel,
    VIF,
    final_data,
)

RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_dataset():
    """Imbalanced binary classification data (~15% positive class)."""
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.85, 0.15],   # imbalanced on purpose
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    return X, y


def split_and_scale(X, y):
    """Stratified split (keeps the class ratio) and standardise the features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    print(f"Train: {X_train.shape}, positive rate = {y_train.mean():.3f}")
    print(f"Test : {X_test.shape},  positive rate = {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def select_features(X_train, y_train):
    """Run RFE, inspect VIFs, then backward-eliminate on p-value and VIF."""
    section("Feature selection: Recursive Feature Elimination")
    top_columns = list(RfeClass(X_train, y_train, number_of_features=8).get_rfe_output())
    print(f"RFE-selected columns ({len(top_columns)}): {top_columns}")

    section("Feature selection: Variance Inflation Factor")
    print(VIF(X_train[top_columns]).get_vif_values().to_string(index=False))

    section("Feature selection: backward elimination (p-value + VIF)")
    final_features = final_data(X_train[top_columns], y_train)
    print(f"\nFinal features ({len(final_features)}): {final_features}")
    return final_features


def fit_models(X_train, y_train, features):
    """Fit all three models on the selected features and return them."""
    Xtr = X_train[features]

    section("Model 1: scikit-learn LogisticRegression")
    sk = SkLearnLogisticModel(Xtr, y_train)
    sk.build_model()
    print(sk.get_parameters().to_string(index=False))

    section("Model 2: statsmodels Binomial GLM")
    sm_model = SMLogisticModel(Xtr, y_train)
    sm_model.build_model()
    print(sm_model.summary())

    section("Model 3: from-scratch gradient descent")
    manual = LogisticRegressionModel(Xtr, y_train, learning_rate=0.1, n_iterations=5000)
    manual.build_model()
    print(manual.get_parameters().to_string(index=False))

    return sk, sm_model, manual


def compare_coefficients(sk, sm_model, manual, features):
    """Line up the three models' coefficients in one table."""
    section("Side-by-side coefficient comparison")
    ordered = ["Intercept"] + features

    def coefs(model):
        params = model.get_parameters().set_index("Feature")
        return params.reindex(ordered)["Coefficient"].values

    comparison = pd.DataFrame({
        "Feature": ordered,
        "sklearn": coefs(sk),
        "statsmodels": coefs(sm_model),
        "Manual GD": coefs(manual),
    })
    print(comparison.to_string(index=False))


def evaluate(sk, X_test, y_test, features, show_plots):
    """Full metric suite, threshold sweep, and the balanced-weight comparison."""
    Xte = X_test[features]
    y_proba = sk.predict_proba(Xte)
    y_pred = sk.predict(Xte)

    section("Classification metrics (default 0.5 threshold)")
    metrics = ClassificationMetrics(y_test, y_pred, y_proba)
    metrics.get_metrics()

    section("Threshold sweep (accuracy / sensitivity / specificity)")
    print(metrics.cutoff_table(plot=show_plots).to_string(index=False))

    section("Effect of class_weight='balanced'")
    balanced = SkLearnLogisticModel(sk.X_train, sk.y_train, class_weight="balanced")
    balanced.build_model()
    print("Default model:")
    metrics.get_metrics()
    print("\nWith class_weight='balanced':")
    ClassificationMetrics(
        y_test, balanced.predict(Xte), balanced.predict_proba(Xte)
    ).get_metrics()

    if show_plots:
        section("Diagnostic plots")
        metrics.plot_confusion_matrix()
        metrics.plot_roc_curve()
        metrics.plot_precision_recall_curve()


def main():
    parser = argparse.ArgumentParser(description="Run the logistic regression walkthrough.")
    parser.add_argument(
        "--plot", action="store_true", help="show the diagnostic plots"
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    section("Dataset")
    X, y = build_dataset()

    section("Train / test split + scaling")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    features = select_features(X_train, y_train)
    sk, sm_model, manual = fit_models(X_train, y_train, features)
    compare_coefficients(sk, sm_model, manual, features)
    evaluate(sk, X_test, y_test, features, show_plots=args.plot)

    print()


if __name__ == "__main__":
    main()
