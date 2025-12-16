import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from backend.utils import ensure_data_dir, get_feature_engineered_path, get_model_path

FEATURES = [
    'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
    'channel_encoded', 'deviation_from_avg', 'amount_to_max_ratio', 'rolling_std',
    'hour', 'day_of_week', 'is_weekend', 'is_night',
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
    'intl_ratio', 'time_since_last', 'recent_burst', 'txn_count_10min', 'txn_count_1hour'
]

def train_model():
    ensure_data_dir()
    df = pd.read_csv(get_feature_engineered_path())
    
    available = [c for c in FEATURES if c in df.columns]
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X)
    
    with open(get_model_path(), 'wb') as f:
        pickle.dump({'model': model, 'features': available}, f)
    
    preds = model.predict(X)
    print(f"Trained. Anomalies: {(preds == -1).sum()} ({100*(preds == -1).mean():.2f}%)")
    return model, available

def load_model():
    with open(get_model_path(), 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']

if __name__ == "__main__":
    train_model()
