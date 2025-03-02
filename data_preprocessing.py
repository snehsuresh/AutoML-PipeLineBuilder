import os
import kaggle
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from kaggle.rest import ApiException

def download_data():
    os.makedirs("data", exist_ok=True)
    try:
        print("Downloading dataset from Kaggle...")
        kaggle.api.competition_download_file('GiveMeSomeCredit', 'cs-training.csv', path='data')
    except ApiException as e:
        if e.status == 403:
            print("\n🚨 Kaggle API returned 403 Forbidden.")
            print("👉 You must manually accept the competition rules before downloading data.")
            print("1. Go to: https://www.kaggle.com/competitions/GiveMeSomeCredit/rules")
            print("2. Log into Kaggle and click 'I Understand and Accept'.")
            print("3. Re-run this script.")
            exit(1)
        else:
            raise e

def load_data():
    file_path = 'data/cs-training.csv'
    if not os.path.exists(file_path):
        download_data()
    df = pd.read_csv(file_path)
    return df

def main():
    df = load_data()
    df.rename(columns={'SeriousDlqin2yrs': 'default'}, inplace=True)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df.dropna(inplace=True)
    df['MonthlyIncome_log'] = np.log1p(df['MonthlyIncome'])
    df['age'] = df['age'].astype(int)
    df.to_csv('cleaned_credit_data.csv', index=False)
    profile = ProfileReport(df, title="Credit Risk Data Profiling Report", explorative=True)
    profile.to_file("credit_data_profile.html")
    print("Profiling report saved to 'credit_data_profile.html'")

if __name__ == "__main__":
    main()
