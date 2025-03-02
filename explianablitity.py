#!/usr/bin/env python
"""
Explainability Script
This script loads the trained AutoML pipeline, performs SHAP explainability analysis,
and generates counterfactual explanations using Alibi Explain.
Usage: python explainability.py
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def load_data():
    df = pd.read_csv("cleaned_titanic.csv")
    return df

def load_pipeline():
    pipeline = joblib.load("automl_pipeline.pkl")
    return pipeline

def shap_explain(pipeline, X):
    # Use a copy of X for SHAP (to reduce computation you could sample a subset)
    X_sample = X.copy()
    
    # Transform data using the pipeline’s preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X_sample)
    
    model = pipeline.named_steps['classifier']
    # Use TreeExplainer if model is tree‑based, else fall back to KernelExplainer
    if hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    else:
        explainer = shap.KernelExplainer(pipeline.predict_proba, X_transformed[:100])
        shap_values = explainer.shap_values(X_transformed)
    
    # Plot SHAP summary plot for class 1
    plt.figure()
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_transformed, show=False)
    plt.title("SHAP Summary Plot for Class 1")
    plt.savefig("shap_summary.png", bbox_inches="tight")
    print("SHAP summary plot saved to 'shap_summary.png'")
    plt.close()

def generate_counterfactual(pipeline, X):
    # For counterfactual explanation we work on a numeric representation of the features.
    # We define numerical features and encode categorical features as numeric codes.
    numerical_features = ['age', 'sibsp', 'parch', 'fare', 'pclass']
    categorical_features = ['sex', 'embarked']
    
    # Build mapping dictionaries for categorical features
    cat_mappings = {}
    for col in categorical_features:
        cats = sorted(X[col].unique())
        mapping = {i: cat for i, cat in enumerate(cats)}
        reverse_mapping = {cat: i for i, cat in mapping.items()}
        cat_mappings[col] = {'mapping': mapping, 'reverse': reverse_mapping}
    
    # Create a numeric version of X for counterfactual generation
    X_cat_encoded = X[categorical_features].copy()
    for col in categorical_features:
        X_cat_encoded[col] = X_cat_encoded[col].map(cat_mappings[col]['reverse'])
    X_cf = pd.concat([X[numerical_features], X_cat_encoded], axis=1)
    
    # Define feature range: (min, max) for each feature column
    feature_min = X_cf.min().values
    feature_max = X_cf.max().values
    feature_range = (feature_min, feature_max)
    
    # Select an instance to explain (the first instance)
    instance = X_cf.iloc[[0]].values  # as numpy array
    
    # Define a prediction function that converts numeric inputs back to original format
    def predict_fn(x):
        df_temp = pd.DataFrame(x, columns=X_cf.columns)
        # For categorical features, round and map numeric codes back to original categories
        for col in categorical_features:
            df_temp[col] = df_temp[col].round().astype(int).map(cat_mappings[col]['mapping'])
        return pipeline.predict_proba(df_temp)
    
    from alibi.explainers import Counterfactual
    # Set target_class as the opposite of the current prediction for the selected instance
    current_pred = pipeline.predict(X.iloc[[0]])[0]
    target_class = 1 - current_pred
    
    cf = Counterfactual(predict_fn, shape=(1, X_cf.shape[1]), target_proba=0.5, tol=0.05,
                        target_class=target_class, max_iter=1000, early_stop=50, feature_range=feature_range)
    
    explanation = cf.explain(instance)
    
    # Save counterfactual explanation details to a text file
    with open("counterfactual_explanation.txt", "w") as f:
        f.write(str(explanation.cf))
    print("Counterfactual explanation saved to 'counterfactual_explanation.txt'")

def main():
    df = load_data()
    X = df.drop('survived', axis=1)
    pipeline = load_pipeline()
    
    # Generate and save SHAP explanation plot
    shap_explain(pipeline, X)
    
    # Generate and save counterfactual explanation details
    generate_counterfactual(pipeline, X)
    
if __name__ == "__main__":
    main()
