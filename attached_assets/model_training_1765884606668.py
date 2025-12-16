import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from backend.utils import ensure_data_dir, get_feature_engineered_path, get_model_path

FEATURE_COLUMNS = [
    'transaction_amount', 'flag_amount', 'merchant_type_encoded', 'merchant_risk_score',
    'channel_encoded', 'deviation_from_user_avg', 'amount_to_user_max_ratio', 'rolling_std_amount',
    'transaction_velocity', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_international_ratio',
    'user_high_risk_txn_ratio', 'user_txn_frequency', 'user_multiple_accounts_flag',
    'cross_account_transfer_ratio', 'is_new_beneficiary', 'beneficiary_txn_count_30d',
    'beneficiary_risk_score', 'geo_anomaly_flag', 'recent_activity_burst',
    'time_since_last_txn', 'txn_count_10min', 'txn_count_1hour'
]

def train_model():
    """
    Train Isolation Forest model on feature engineered data.
    Includes velocity features for ML-based frequency anomaly detection.
    """
    ensure_data_dir()
    
    feature_path = get_feature_engineered_path()
    model_path = get_model_path()
    
    print(f"Loading feature engineered data from: {feature_path}")
    df = pd.read_csv(feature_path)
    
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    print(f"Using {len(available_features)} features for training")
    
    X = df[available_features].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Isolation Forest model...")
    model.fit(X)
    
    model_data = {
        'model': model,
        'features': available_features
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path}")
    
    predictions = model.predict(X)
    anomaly_count = (predictions == -1).sum()
    print(f"Training complete. Detected {anomaly_count} anomalies ({100*anomaly_count/len(X):.2f}%)")
    
    return model, available_features

def load_model():
    """Load trained model from disk"""
    model_path = get_model_path()
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['features']

if __name__ == "__main__":
    train_model()
