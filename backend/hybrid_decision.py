import numpy as np
from datetime import datetime
from backend.rule_engine import check_rule_violation
from backend.utils import TRANSFER_TYPE_ENCODED, TRANSFER_TYPE_RISK

def prepare_features(txn, user_stats):
    f = {}
    f['transaction_amount'] = txn.get('amount', 0)
    t_type = str(txn.get('transfer_type', 'O')).upper()
    f['flag_amount'] = 1 if t_type == 'S' else 0
    f['transfer_type_encoded'] = TRANSFER_TYPE_ENCODED.get(t_type, 0)
    f['transfer_type_risk'] = TRANSFER_TYPE_RISK.get(t_type, 0.5)
    f['channel_encoded'] = 0
    f['user_avg_amount'] = user_stats.get('user_avg_amount', 0)
    f['user_std_amount'] = user_stats.get('user_std_amount', 0)
    f['user_max_amount'] = user_stats.get('user_max_amount', 0)
    f['user_txn_frequency'] = user_stats.get('user_txn_frequency', 0)
    f['deviation_from_avg'] = abs(f['transaction_amount'] - f['user_avg_amount'])
    f['amount_to_max_ratio'] = f['transaction_amount'] / f['user_max_amount'] if f['user_max_amount'] > 0 else 1
    f['rolling_std'] = f['user_std_amount']
    now = datetime.now()
    f['hour'], f['day_of_week'] = now.hour, now.weekday()
    f['is_weekend'] = 1 if f['day_of_week'] >= 5 else 0
    f['is_night'] = 1 if f['hour'] < 6 or f['hour'] >= 22 else 0
    f['intl_ratio'] = 0.1 if t_type == 'S' else 0
    f['time_since_last'] = txn.get('time_since_last_txn', 3600)
    f['txn_count_10min'] = txn.get('txn_count_10min', 1)
    f['txn_count_1hour'] = txn.get('txn_count_1hour', 1)
    f['recent_burst'] = 1 if f['time_since_last'] < 300 else 0
    return f

def make_decision(txn, user_stats, model=None, features_list=None):
    result = {'is_fraud': False, 'ml_prediction': 1, 'reasons': [], 'risk_score': 0.0, 'threshold': 0.0, 'velocity_anomaly': False}
    txn_count = txn.get('txn_count_10min', 1)
    ml_anomaly = False
    
    if model and features_list:
        try:
            f = prepare_features(txn, user_stats)
            vec = np.array([[f.get(c, 0) for c in features_list]])
            vec = np.nan_to_num(vec, nan=0, posinf=0, neginf=0)
            pred = model.predict(vec)[0]
            result['ml_prediction'] = pred
            result['risk_score'] = -model.decision_function(vec)[0]
            if pred == -1:
                ml_anomaly = True
                result['velocity_anomaly'] = True
                result['reasons'].append(f"ML detected anomaly (velocity: {txn_count} txns, risk: {result['risk_score']:.4f})")
        except Exception as e:
            print(f"ML error: {e}")
    
    violated, reason, threshold = check_rule_violation(
        txn.get('amount', 0), user_stats.get('user_avg_amount', 0),
        user_stats.get('user_std_amount', 0), txn.get('transfer_type', 'O'),
        ml_anomaly, txn_count
    )
    result['threshold'] = threshold
    if violated and reason not in result['reasons']:
        result['reasons'].append(reason)
    
    result['is_fraud'] = result['ml_prediction'] == -1 or violated
    return result
