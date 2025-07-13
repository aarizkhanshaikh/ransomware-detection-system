import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_dataframe(df):
    # Drop columns with >90% missing values
    threshold = 0.9
    df = df.loc[:, df.isnull().mean() < threshold]

    # Convert all columns to numeric if possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Encode object-type columns if any (after coercion to numeric, may still exist)
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Handle NaN and inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

# Load datasets
benign_df = pd.read_csv("benign_data.csv")
ransomware_df = pd.read_csv("ransomware_data.csv")

# Clean datasets
clean_benign_df = clean_dataframe(benign_df)
clean_ransomware_df = clean_dataframe(ransomware_df)

# Save cleaned datasets
clean_benign_df.to_csv("clean_benign_data.csv", index=False)
clean_ransomware_df.to_csv("clean_ransomware_data.csv", index=False)

print("Datasets cleaned and saved as 'clean_benign_data.csv' and 'clean_ransomware_data.csv'")