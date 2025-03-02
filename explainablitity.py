#!/usr/bin/env python
"""
AutoML Pipeline Script for Credit Risk Prediction (Enhanced)
This builds a modular AutoML pipeline for the 'Give Me Some Credit' dataset.
- Advanced preprocessing, log-transforms, PCA (optional), SMOTE for imbalance
- Model selection & tuning using Optuna
- Supports: Logistic Regression, Random Forest, XGBoost, Stacking Ensemble
- Outputs logs to 'automl_pipeline.log'

Usage:
    python automl_pipeline.py
"""

import pandas as pd
import joblib
import optuna
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# --- Logging Configuration ---
logging.basicConfig(
    filename='automl_pipeline.log',
    filemode='w',
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)

# --- Constants ---
FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
    'MonthlyIncome_log', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
TARGET = 'default'

# --- Load Data ---
def load_data():
    return pd.read_csv("cleaned_credit_data.csv")

# --- Preprocessing Pipeline ---
def get_preprocessor():
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer(transformers=[
        ('num', numerical_pipeline, FEATURES)
    ])

# --- Objective Function for Optuna ---
def objective(trial):
    df = load_data()
    X, y = df[FEATURES], df[TARGET]

    preprocessor = get_preprocessor()

    # Optional PCA
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    pca = PCA(n_components=trial.suggest_int("pca_components", 2, len(FEATURES))) if use_pca else None

    # Model Selection
    model_type = trial.suggest_categorical("model_type", ["lr", "rf", "xgb", "stack"])

    if model_type == "lr":
        model = LogisticRegression(
            C=trial.suggest_float("lr_C", 1e-4, 1e2, log=True),
            solver='liblinear', max_iter=500, random_state=42
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 300),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            random_state=42
        )
    elif model_type == "xgb":
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 50, 300),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
            learning_rate=trial.suggest_float("xgb_learning_rate", 1e-3, 1.0, log=True),
            eval_metric='logloss', tree_method='gpu_hist', random_state=42
        )
    elif model_type == "stack":
        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("stack_rf_estimators", 50, 200), 
            random_state=42
        )
        xgb_clf = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("stack_xgb_estimators", 50, 200),
            learning_rate=trial.suggest_float("stack_xgb_lr", 1e-3, 1.0, log=True),
            eval_metric='logloss', tree_method='gpu_hist', random_state=42
        )
        final_estimator = LogisticRegression(
            C=trial.suggest_float("stack_meta_lr_C", 1e-4, 1e2, log=True),
            solver='liblinear', max_iter=500, random_state=42
        )
        model = StackingClassifier(
            estimators=[('rf', rf), ('xgb', xgb_clf)],
            final_estimator=final_estimator, n_jobs=-1, cv=5
        )
    else:
        raise ValueError("Unknown model type")

    # Pipeline: Preprocessor -> (Optional PCA) -> SMOTE -> Model
    steps = [('preprocessor', preprocessor)]
    if pca: steps.append(('pca', pca))
    steps += [('smote', SMOTE(random_state=42)), ('classifier', model)]
    pipeline = ImbPipeline(steps)

    # Evaluate with Stratified 5-Fold Cross Validation (Parallelized)
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)

    return score.mean()

# --- Main Training Function ---
def main():
    logging.info("Starting AutoML Pipeline")
    study = optuna.create_study(study_name="credit_risk_automl", direction="maximize")

    with tqdm(total=100, desc="Optimizing Hyperparameters") as pbar:
        def callback(study, trial):
            pbar.update(1)
            logging.info(f"Trial {trial.number} finished with value: {trial.value:.4f} and params: {trial.params}")
        
        study.optimize(objective, n_trials=100, callbacks=[callback])

    # Log Best Result
    trial = study.best_trial
    logging.info(f"Best trial achieved ROC AUC: {trial.value:.4f}")
    logging.info(f"Best parameters: {trial.params}")

    # Train Final Model on Full Dataset
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    preprocessor = get_preprocessor()

    # Rebuild Final Pipeline Based on Best Params
    pca = PCA(n_components=trial.params["pca_components"]) if trial.params.get("use_pca", False) else None

    if trial.params["model_type"] == "lr":
        model = LogisticRegression(C=trial.params["lr_C"], solver='liblinear', max_iter=500, random_state=42)
    elif trial.params["model_type"] == "rf":
        model = RandomForestClassifier(n_estimators=trial.params["rf_n_estimators"], 
                                       max_depth=trial.params["rf_max_depth"], random_state=42)
    elif trial.params["model_type"] == "xgb":
        model = xgb.XGBClassifier(n_estimators=trial.params["xgb_n_estimators"],
                                  max_depth=trial.params["xgb_max_depth"],
                                  learning_rate=trial.params["xgb_learning_rate"],
                                  eval_metric='logloss', tree_method='gpu_hist', random_state=42)
    elif trial.params["model_type"] == "stack":
        rf = RandomForestClassifier(n_estimators=trial.params["stack_rf_estimators"], random_state=42)
        xgb_clf = xgb.XGBClassifier(n_estimators=trial.params["stack_xgb_estimators"],
                                    learning_rate=trial.params["stack_xgb_lr"],
                                    eval_metric='logloss', tree_method='gpu_hist', random_state=42)
        final_estimator = LogisticRegression(C=trial.params["stack_meta_lr_C"], solver='liblinear', max_iter=500, random_state=42)
        model = StackingClassifier(estimators=[('rf', rf), ('xgb', xgb_clf)], final_estimator=final_estimator, n_jobs=-1, cv=5)

    steps = [('preprocessor', preprocessor)]
    if pca: steps.append(('pca', pca))
    steps += [('smote', SMOTE(random_state=42)), ('classifier', model)]
    final_pipeline = ImbPipeline(steps)

    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, 'automl_pipeline.pkl')
    logging.info("Final model saved to 'automl_pipeline.pkl'")

if __name__ == "__main__":
    main()
