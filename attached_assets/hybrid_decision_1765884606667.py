import pandas as pd
import numpy as np
import pickle
from backend.model_training import load_model, FEATURE_COLUMNS
from backend.rule_engine import check_rule_violation, get_user_stats
from backend.utils import get_feature_engineered_path, MERCHANT_TYPE_ENCODED, MERCHANT_RISK_SCORES

def prepare_transaction_features(transaction_data, user_stats, all_data=None):
    """
    Prepare feature vector for a single transaction.
    Includes velocity features for ML-based frequency detection.
    """
    features = {}
    features['transaction_amount'] = transaction_data.get('amount', 0)
    transfer_type = str(transaction_data.get('transfer_type', 'O')).upper()
    features['flag_amount'] = 1 if transfer_type == 'S' else 0
    features['merchant_type_encoded'] = MERCHANT_TYPE_ENCODED.get(transfer_type, 0)
    features['merchant_risk_score'] = MERCHANT_RISK_SCORES.get(transfer_type, 0.5)
    features['channel_encoded'] = 0
    features['user_avg_amount'] = user_stats.get('user_avg_amount', 0)
    features['user_std_amount'] = user_stats.get('user_std_amount', 0)
    features['user_max_amount'] = user_stats.get('user_max_amount', 0)
    features['user_txn_frequency'] = user_stats.get('user_txn_frequency', 0)
    features['deviation_from_user_avg'] = abs(features['transaction_amount'] - features['user_avg_amount'])
    features['amount_to_user_max_ratio'] = (features['transaction_amount'] / features['user_max_amount'] if features['user_max_amount'] > 0 else 1)
    features['rolling_std_amount'] = features['user_std_amount']
    features['transaction_velocity'] = features['user_txn_frequency']
    
    from datetime import datetime
    now = datetime.now()
    features['hour'] = now.hour
    features['day_of_week'] = now.weekday()
    features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
    features['is_night'] = 1 if features['hour'] < 6 or features['hour'] >= 22 else 0
    features['user_international_ratio'] = 0.1 if transfer_type == 'S' else 0
    features['user_high_risk_txn_ratio'] = 0.1 if transfer_type in ['S', 'Q'] else 0
    features['user_multiple_accounts_flag'] = 0
    features['cross_account_transfer_ratio'] = 0
    features['is_new_beneficiary'] = 0
    features['beneficiary_txn_count_30d'] = 5
    features['beneficiary_risk_score'] = 0.5
    
    features['time_since_last_txn'] = transaction_data.get('time_since_last_txn', 3600)
    features['txn_count_10min'] = transaction_data.get('txn_count_10min', 1)
    features['txn_count_1hour'] = transaction_data.get('txn_count_1hour', 1)
    
    bank_country = str(transaction_data.get('bank_country', 'UAE')).upper()
    local_countries = ['UAE', 'AE', 'UNITED ARAB EMIRATES', 'AJMAN']
    features['geo_anomaly_flag'] = 0 if bank_country in local_countries else 1
    features['recent_activity_burst'] = 1 if features['time_since_last_txn'] < 300 else 0
    
    return features

def make_hybrid_decision(transaction_data, user_stats, model=None, features_list=None):
    """
    Make fraud detection decision combining ML model and rule engine.
    ML model determines velocity anomalies - no hardcoded limits.
    
    Returns:
        dict: {
            'is_fraud': bool,
            'ml_prediction': int (1=normal, -1=anomaly),
            'rule_violation': bool,
            'reasons': list of str,
            'risk_score': float,
            'threshold': float,
            'velocity_anomaly': bool
        }
    """
    result = {
        'is_fraud': False, 
        'ml_prediction': 1, 
        'rule_violation': False,
        'reasons': [], 
        'risk_score': 0.0, 
        'threshold': 0.0,
        'velocity_anomaly': False
    }
    
    window_txn_count = transaction_data.get('txn_count_10min', 1)
    ml_velocity_anomaly = False
    
    if model is not None and features_list is not None:
        try:
            features = prepare_transaction_features(transaction_data, user_stats)
            feature_vector = []
            for col in features_list:
                feature_vector.append(features.get(col, 0))
            feature_array = np.array([feature_vector])
            feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
            
            ml_prediction = model.predict(feature_array)[0]
            result['ml_prediction'] = ml_prediction
            
            anomaly_score = model.decision_function(feature_array)[0]
            result['risk_score'] = -anomaly_score
            
            if ml_prediction == -1:
                ml_velocity_anomaly = True
                result['velocity_anomaly'] = True
                result['reasons'].append(f"ML Model detected anomaly (velocity: {window_txn_count} txns in 10 min, risk score: {result['risk_score']:.4f})")
                    
        except Exception as e:
            print(f"ML prediction error: {e}")
            result['ml_prediction'] = 1
    
    rule_violated, rule_reason, threshold = check_rule_violation(
        transaction_amount=transaction_data.get('amount', 0),
        user_avg_amount=user_stats.get('user_avg_amount', 0),
        user_std_amount=user_stats.get('user_std_amount', 0),
        transfer_type=transaction_data.get('transfer_type', 'O'),
        window_txn_count=window_txn_count,
        ml_velocity_anomaly=ml_velocity_anomaly
    )
    
    result['rule_violation'] = rule_violated
    result['threshold'] = threshold
    if rule_violated and rule_reason not in result['reasons']:
        result['reasons'].append(rule_reason)
    
    result['is_fraud'] = result['ml_prediction'] == -1 or result['rule_violation']
    
    return result
