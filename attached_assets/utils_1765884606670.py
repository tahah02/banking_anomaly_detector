import os
import pandas as pd
import numpy as np

def ensure_data_dir():
    """Ensure data directory exists"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def get_raw_xlsx_path():
    """Get path to raw xlsx file"""
    return 'attached_assets/raw_datav1_1765522826258.xlsx'

def get_raw_csv_path():
    """Get path to raw csv file"""
    return 'data/raw_data.csv'

def get_clean_csv_path():
    """Get path to clean csv file"""
    return 'data/Clean.csv'

def get_feature_engineered_path():
    """Get path to feature engineered csv file"""
    return 'data/feature_Engineered.csv'

def get_model_path():
    """Get path to trained model"""
    return 'models/isolation_forest.pkl'

MERCHANT_TYPE_MAPPING = {
    'S': 'Overseas',
    'I': 'Ajman',
    'L': 'UAE',
    'Q': 'Quick',
    'O': 'Own'
}

MERCHANT_TYPE_ENCODED = {
    'S': 4,
    'I': 1,
    'L': 2,
    'Q': 3,
    'O': 0
}

MERCHANT_RISK_SCORES = {
    'S': 0.9,
    'I': 0.1,
    'L': 0.2,
    'Q': 0.5,
    'O': 0.0
}
