import os

def ensure_data_dir():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def get_clean_csv_path():
    return 'data/Clean.csv'

def get_feature_engineered_path():
    return 'data/feature_Engineered.csv'

def get_model_path():
    return 'models/isolation_forest.pkl'

TRANSFER_TYPE_MAPPING = {'S': 'Overseas', 'I': 'Ajman', 'L': 'UAE', 'Q': 'Quick', 'O': 'Own'}
TRANSFER_TYPE_ENCODED = {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}
TRANSFER_TYPE_RISK = {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}
