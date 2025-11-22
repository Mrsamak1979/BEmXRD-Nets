
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models_saved')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')

# ชื่อไฟล์ CSV ของคุณ
DATA_FILENAME = 'df_ABCDE.csv' 

# Hyperparameters
SEED = 42
KRR_PARAMS = {'alpha': 0.1826, 'degree': 3, 'gamma': 0.0039, 'kernel': 'poly'}
SVR_PARAMS = {'C': 2.8411, 'degree': 4, 'epsilon': 0.0062, 'kernel': 'rbf'}
LGBM_PARAMS = {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1}
MLP_PARAMS = {'hidden_layer_sizes': (300, 57, 30), 'max_iter': 500, 'random_state': 42}
