import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os


DATA_PATH = 'data/feature_Engineered_3.csv' 
MODEL_PATH = 'backend/isolation_forest.pkl'
FEATURES = [
    'transaction_amount', 
    'flag_amount',             
    'transfer_type_encoded',   
    'transfer_type_risk',      
    'channel_encoded', 
    'hour', 
    'day_of_week', 
    'is_weekend', 
    'is_night', 
    'user_avg_amount', 
    'user_std_amount', 
    'user_max_amount', 
    'user_txn_frequency', 
    'deviation_from_avg',      
    'amount_to_max_ratio',     
    'intl_ratio',             
    'time_since_last',         
    # 'recent_burst',            
    'rolling_std',            
    'txn_count_10min', 
    'txn_count_30s'            
]

def load_data():
    paths_to_try = [
        DATA_PATH,
        'feature_Engineered(1).csv',
        'data/feature_Engineered(1).csv'
    ]   
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)          
    print("Error: feature_Engineered(1).csv not found! Please upload it to the 'data' folder.")
    return None

def train_model():
    print("Initializing Model Training...")
    df = load_data() 
    if df is None:
        return
    X = df[FEATURES].fillna(0)   
    print(f"Training on {len(FEATURES)} features.")
    print(f"Sample Features: {FEATURES[:5]}...")

    # Train Isolation Forest
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X)
    model_data = {
        'model': clf,
        'features': FEATURES
    } 
    joblib.dump(model_data, MODEL_PATH)
    print(f"SUCCESS: Model saved to {MODEL_PATH}")
    print("Training Complete!")

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            if isinstance(data, dict):
                return data['model'], data['features']
            else:
                return data, FEATURES 
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Retraining model...")
            train_model()
            return load_model()
    else:
        print("Model file not found. Training new model...")
        train_model()
        return load_model()

if __name__ == "__main__":
    train_model()