#!/usr/bin/env python
"""
AutoML Pipeline Script
This script builds a modular AutoML pipeline for the Titanic dataset.
It performs preprocessing, feature engineering, model selection, and hyperparameter tuning using Optuna.
Usage: python automl_pipeline.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("cleaned_titanic.csv")
    return df

def get_preprocessor(numerical_features, categorical_features):
    # Pipeline for numerical features: imputation and scaling
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features: imputation and one-hot encoding
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor

def objective(trial):
    df = load_data()
    # Define features and target
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    # Define numerical and categorical columns
    numerical_features = ['age', 'sibsp', 'parch', 'fare', 'pclass']
    categorical_features = ['sex', 'embarked']
    
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    # Select model via hyperparameter choice
    model_name = trial.suggest_categorical('model', ['lr', 'rf', 'xgb'])
    
    if model_name == 'lr':
        C = trial.suggest_loguniform('lr_C', 1e-4, 1e2)
        model = LogisticRegression(C=C, solver='liblinear', max_iter=200)
    elif model_name == 'rf':
        n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
        max_depth = trial.suggest_int('rf_max_depth', 3, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == 'xgb':
        n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
        max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
        learning_rate = trial.suggest_loguniform('xgb_learning_rate', 1e-3, 1.0)
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss',
                                  tree_method='gpu_hist')  # utilize GPU if available
    else:
        raise ValueError("Invalid model selected.")
    
    # Create complete pipeline: preprocessing + classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Evaluate with 5-fold cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {:.4f}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Train final model on full data using the best hyperparameters
    df = load_data()
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    numerical_features = ['age', 'sibsp', 'parch', 'fare', 'pclass']
    categorical_features = ['sex', 'embarked']
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    best_model_name = trial.params['model']
    if best_model_name == 'lr':
        model = LogisticRegression(C=trial.params['lr_C'], solver='liblinear', max_iter=200)
    elif best_model_name == 'rf':
        model = RandomForestClassifier(n_estimators=trial.params['rf_n_estimators'],
                                       max_depth=trial.params['rf_max_depth'],
                                       random_state=42)
    elif best_model_name == 'xgb':
        model = xgb.XGBClassifier(n_estimators=trial.params['xgb_n_estimators'],
                                  max_depth=trial.params['xgb_max_depth'],
                                  learning_rate=trial.params['xgb_learning_rate'],
                                  use_label_encoder=False, eval_metric='logloss',
                                  tree_method='gpu_hist')
    else:
        raise ValueError("Invalid model selected.")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y, y_pred)
    print("Final model accuracy on full dataset: {:.4f}".format(acc))
    
    # Save the complete pipeline for later use
    joblib.dump(pipeline, 'automl_pipeline.pkl')
    print("Trained pipeline saved to 'automl_pipeline.pkl'")

if __name__ == "__main__":
    main()
