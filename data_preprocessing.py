#!/usr/bin/env python
"""
Data Preprocessing Script
This script loads the Titanic dataset from seaborn, performs basic data cleaning,
generates an interactive Pandas Profiling report, and saves the cleaned data to a CSV file.
Usage: python data_preprocessing.py
"""

import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport

def main():
    # Load Titanic dataset from seaborn
    df = sns.load_dataset('titanic')
    
    # Select relevant columns
    df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
    
    # Basic cleaning: fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # Save cleaned data
    df.to_csv('cleaned_titanic.csv', index=False)
    print("Cleaned data saved to 'cleaned_titanic.csv'")
    
    # Generate Pandas Profiling Report
    profile = ProfileReport(df, title="Titanic Data Profiling Report", explorative=True)
    profile.to_file("titanic_data_profile.html")
    print("Pandas Profiling report saved to 'titanic_data_profile.html'")

if __name__ == "__main__":
    main()
