#!/usr/bin/env python
"""
Explainability Script for Credit Risk Prediction
Loads the trained AutoML pipeline, performs SHAP explainability analysis,
and generates counterfactual explanations using DiCE.
Usage:
    python explainability.py
"""

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import dice_ml
from dice_ml import Dice
from sklearn.ensemble import RandomForestClassifier
# shap.initjs()
# Features used in training — must match automl_pipeline.py
FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
    'MonthlyIncome_log', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
TARGET = 'default'

# --- Load Data and Pipeline ---
def load_data():
    return pd.read_csv("cleaned_credit_data.csv")

def load_pipeline():
    return joblib.load("automl_pipeline.pkl")

# --- SHAP Explainability ---
def shap_explain(pipeline, X):
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']

    # Preprocess input data
    X_transformed = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())

    def predict_proba_wrapper(X):
        return model.predict_proba(X)

    if hasattr(model, 'get_booster') or isinstance(model, RandomForestClassifier):
        # Fast TreeExplainer -> Use full data
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        shap_X = X_transformed  # Use full data
    else:
        # Slow KernelExplainer -> Use sampled data
        background_sample = X_transformed.sample(500, random_state=42)
        explainer = shap.KernelExplainer(predict_proba_wrapper, background_sample)

        shap_X = X_transformed.sample(100, random_state=42)
        shap_values = explainer.shap_values(shap_X)

    plt.figure()
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, shap_X, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("shap_summary.png", bbox_inches="tight")
    print("SHAP summary plot saved to 'shap_summary.png'")
    plt.close()




# --- Counterfactual Explanation using DiCE ---
def generate_counterfactual(pipeline, X, y):
    # Prepare data in same format used for training (features + target)
    data = pd.concat([X, y], axis=1)

    # Create Data object for DiCE
    d = dice_ml.Data(
        dataframe=data,
        continuous_features=FEATURES,
        outcome_name=TARGET
    )

    # Define wrapper for sklearn pipeline to make it work with DiCE
    class PipelineWrapper:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def predict_proba(self, df):
            return self.pipeline.predict_proba(df)

    model_wrapper = PipelineWrapper(pipeline)

    m = dice_ml.Model(model=model_wrapper, backend="sklearn", model_type="classifier")

    exp = Dice(d, m)

    # Select a query instance — first row for now
    query_instance = X.iloc[[0]]

    explanation = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
    if explanation is None or explanation.cf_examples_list[0].final_cfs_df is None:
        print("⚠️ Warning: DiCE could not generate valid counterfactuals. No file saved.")
        return  # Exit gracefully

    # Save to file
    explanation.visualize_as_dataframe().to_csv("counterfactual_explanation.csv", index=False)
    print("Counterfactual explanation saved to 'counterfactual_explanation.csv'")

# --- Main Execution ---
def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]
    pipeline = load_pipeline()

    shap_explain(pipeline, X)
    generate_counterfactual(pipeline, X, y)

if __name__ == "__main__":
    main()
