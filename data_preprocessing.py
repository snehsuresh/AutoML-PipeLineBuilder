#!/usr/bin/env python
"""
Data Preprocessing Script for Credit Risk Dataset
This script downloads the 'Give Me Some Credit' dataset from Kaggle,
performs advanced data cleaning and feature engineering,
generates an interactive profiling report, and saves the cleaned data.
Usage: python data_preprocessing.py
"""

import os
import kaggle
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

def download_data():
    # Make sure the output folder exists
    os.makedirs("data", exist_ok=True)
    
    # Download dataset using Kaggle API
    print("Downloading dataset from Kaggle...")
    kaggle.api.competition_download_file('GiveMeSomeCredit', 'cs-training.csv', path='data')
    
    # Unzip if needed (the file itself is a CSV, so no unzip required here)

def load_data():
    file_path = 'data/cs-training.csv'
    if not os.path.exists(file_path):
        download_data()
    df = pd.read_csv(file_path)
    return df

def main():
    # Load dataset
    df = load_data()
    print("Initial dataset shape:", df.shape)
    
    # Rename columns for convenience
    df.rename(columns={'SeriousDlqin2yrs': 'default'}, inplace=True)

    # --- Data Cleaning & Feature Engineering ---
    # Fill missing MonthlyIncome with median value
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    
    # Drop any remaining missing values (just in case)
    df.dropna(inplace=True)
    
    # Log-transform MonthlyIncome to reduce skewness
    df['MonthlyIncome_log'] = np.log1p(df['MonthlyIncome'])
    
    # Convert age to integer (some minor floats can appear due to weird CSV formatting)
    df['age'] = df['age'].astype(int)
    
    # Save cleaned data
    df.to_csv('cleaned_credit_data.csv', index=False)
    print("Cleaned data saved to 'cleaned_credit_data.csv'")
    
    # Generate an interactive profiling report
    profile = ProfileReport(df, title="Credit Risk Data Profiling Report", explorative=True)
    profile.to_file("credit_data_profile.html")
    print("Profiling report saved to 'credit_data_profile.html'")

if __name__ == "__main__":
    main()
