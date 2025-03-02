#!/usr/bin/env python
"""
AutoML Pipeline Script for Credit Risk Prediction
This script builds a modular AutoML pipeline for the Credit Risk dataset (Give Me Some Credit).
It performs advanced preprocessing, feature engineering, oversampling (SMOTE), optional PCA,
and model selection/hyperparameter tuning using Optuna.
Models include Logistic Regression, Random Forest, XGBoost, and a Stacking ensemble.
Usage: python automl_pipeline.py
"""

import pandas as pd
import joblib
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def load_data():
    df = pd.read_csv("cleaned_credit_data.csv")
    return df

def get_preprocessor(numerical_features):
    # Pipeline for numerical features: imputation and scaling
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_features)
    ])
    return preprocessor

def objective(trial):
    df = load_data()
    # Define features and target.
    # We use the engineered 'MonthlyIncome_log' in place of raw MonthlyIncome.
    features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                'DebtRatio', 'MonthlyIncome_log', 'NumberOfOpenCreditLinesAndLoans',
                'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse']
    X = df[features]
    y = df['default']
    
    numerical_features = features

    preprocessor = get_preprocessor(numerical_features)
    
    # Optional PCA for dimensionality reduction
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    if use_pca:
        n_components = trial.suggest_int("pca_components", 2, len(numerical_features))
        pca = PCA(n_components=n_components, random_state=42)
    else:
        pca = None

    # Oversample minority class using SMOTE
    smote = SMOTE(random_state=42)

    # Select model type
    model_type = trial.suggest_categorical("model_type", ["lr", "rf", "xgb", "stack"])
    
    if model_type == "lr":
        C = trial.suggest_loguniform("lr_C", 1e-4, 1e2)
        model = LogisticRegression(C=C, solver='liblinear', max_iter=500, random_state=42)
    elif model_type == "rf":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        max_depth = trial.suggest_int("rf_max_depth", 3, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "xgb":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        learning_rate = trial.suggest_loguniform("xgb_learning_rate", 1e-3, 1.0)
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss',
                                  tree_method='gpu_hist', random_state=42)
    elif model_type == "stack":
        # Define base estimators for stacking
        n_estimators_rf = trial.suggest_int("stack_rf_estimators", 50, 200)
        rf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=42)
        n_estimators_xgb = trial.suggest_int("stack_xgb_estimators", 50, 200)
        learning_rate_stack = trial.suggest_loguniform("stack_xgb_lr", 1e-3, 1.0)
        xgb_clf = xgb.XGBClassifier(n_estimators=n_estimators_xgb, use_label_encoder=False, eval_metric='logloss',
                                    tree_method='gpu_hist', learning_rate=learning_rate_stack, random_state=42)
        estimators = [
            ('rf', rf),
            ('xgb', xgb_clf)
        ]
        # Meta learner
        meta_lr_C = trial.suggest_loguniform("stack_meta_lr_C", 1e-4, 1e2)
        final_estimator = LogisticRegression(C=meta_lr_C, solver='liblinear', max_iter=500, random_state=42)
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, cv=5)
    else:
        raise ValueError("Invalid model_type selected.")
    
    # Build the complete pipeline
    steps = []
    steps.append(('preprocessor', preprocessor))
    if pca is not None:
        steps.append(('pca', pca))
    steps.append(('smote', smote))
    steps.append(('classifier', model))
    
    pipeline = ImbPipeline(steps=steps)
    
    # Evaluate using stratified 5-fold cross-validation with ROC AUC as metric
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    return scores.mean()

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    print("Best trial:")
    trial = study.best_trial
    print("  ROC AUC: {:.4f}".format(trial.value))
    print("  Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Re-train the final model on the full dataset using the best hyperparameters
    df = load_data()
    features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                'DebtRatio', 'MonthlyIncome_log', 'NumberOfOpenCreditLinesAndLoans',
                'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                'NumberOfTime60-89DaysPastDueNotWorse']
    X = df[features]
    y = df['default']
    
    numerical_features = features
    preprocessor = get_preprocessor(numerical_features)
    
    # Rebuild PCA if used
    if trial.params.get("use_pca", False):
        n_components = trial.params["pca_components"]
        pca = PCA(n_components=n_components, random_state=42)
    else:
        pca = None
    
    # Rebuild best classifier
    model_type = trial.params["model_type"]
    if model_type == "lr":
        model = LogisticRegression(C=trial.params["lr_C"], solver='liblinear', max_iter=500, random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=trial.params["rf_n_estimators"],
                                       max_depth=trial.params["rf_max_depth"], random_state=42)
    elif model_type == "xgb":
        model = xgb.XGBClassifier(n_estimators=trial.params["xgb_n_estimators"],
                                  max_depth=trial.params["xgb_max_depth"],
                                  learning_rate=trial.params["xgb_learning_rate"],
                                  use_label_encoder=False, eval_metric='logloss',
                                  tree_method='gpu_hist', random_state=42)
    elif model_type == "stack":
        rf = RandomForestClassifier(n_estimators=trial.params["stack_rf_estimators"], random_state=42)
        xgb_clf = xgb.XGBClassifier(n_estimators=trial.params["stack_xgb_estimators"],
                                    learning_rate=trial.params["stack_xgb_lr"],
                                    use_label_encoder=False, eval_metric='logloss',
                                    tree_method='gpu_hist', random_state=42)
        estimators = [
            ('rf', rf),
            ('xgb', xgb_clf)
        ]
        final_estimator = LogisticRegression(C=trial.params["stack_meta_lr_C"], solver='liblinear', max_iter=500, random_state=42)
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, cv=5)
    else:
        raise ValueError("Invalid model_type selected.")
    
    # Build final pipeline steps
    from imblearn.over_sampling import SMOTE  # re-import in case
    steps = []
    steps.append(('preprocessor', preprocessor))
    if pca is not None:
        steps.append(('pca', pca))
    steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('classifier', model))
    
    final_pipeline = ImbPipeline(steps=steps)
    
    # Fit the final pipeline on the entire dataset
    final_pipeline.fit(X, y)
    y_pred_proba = final_pipeline.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_pred_proba)
    print("Final model ROC AUC on full dataset: {:.4f}".format(auc))
    
    # Save the trained pipeline
    joblib.dump(final_pipeline, "automl_pipeline.pkl")
    print("Trained pipeline saved to 'automl_pipeline.pkl'")

if __name__ == "__main__":
    main()
