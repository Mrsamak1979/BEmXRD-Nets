import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import warnings # <--- เพิ่ม import

# --- ปิด Warning ---
warnings.filterwarnings("ignore")
# ------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.config import MODEL_SAVE_PATH, RESULTS_PATH, SEED
from src.data_loader import load_data, preprocess_data
from src.models import ModelFactory

def main():
    print("--- 🚀 BEmXRD-Nets v2 Started ---")
    
    # 1. Load & Process
    try:
        df = load_data()
        X, y = preprocess_data(df, target_col='energy_per_atom')
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Split
    print(f"✂️ Splitting Data (Shape: {X.shape})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # 3. Train
    print("🏋️ Training Stacking Model...")
    model = ModelFactory.get_stacking_model()
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{'='*30}\n   R2 Score: {r2:.4f}\n   MAE:      {mae:.4f}\n   MSE:      {mse:.4f}\n{'='*30}")
    
    # 5. Save
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, 'bemxrd_v2_model.pkl'))
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(f'Parity Plot (R2={r2:.2f})')
    plt.savefig(os.path.join(RESULTS_PATH, 'parity_plot_v2.png'))
    print(f"📊 Plot saved to results/")

if __name__ == "__main__":
    main()
