
import pandas as pd
import numpy as np
import os
from src.config import DATA_RAW_PATH, DATA_FILENAME
from src.features import generate_advanced_atomic_features
from src.xrd_pipeline import create_df_xrd_final

def load_data():
    path = os.path.join(DATA_RAW_PATH, DATA_FILENAME)
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def preprocess_data(df, target_col='energy_per_atom'):
    print(f"--- Preprocessing (Samples: {len(df)}) ---")
    
    # 1. Atomic Features
    df_atomic = generate_advanced_atomic_features(df, 'full_formula' if 'full_formula' in df else 'formula')
    
    # 2. XRD Pipeline (Returns Target + PCA Features)
    df_xrd = create_df_xrd_final(df, 'cif', target_col)
    
    # 3. Merge
    df_final = pd.concat([df_xrd.reset_index(drop=True), df_atomic.reset_index(drop=True)], axis=1)
    
    # 4. Clean Non-Numeric & NaN
    print("🧹 Cleaning Data...")
    df_final = df_final.loc[:, ~df_final.columns.duplicated()] # Remove duplicate cols
    df_final = df_final.apply(pd.to_numeric, errors='coerce')
    df_final.dropna(inplace=True)
    
    if target_col in df_final.columns:
        y = df_final[target_col]
        X = df_final.drop(columns=[target_col])
        return X, y
    return df_final, None
