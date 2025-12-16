import pandas as pd
import numpy as np

TRANSFER_TYPE_MULTIPLIERS = {
    'S': 2.0, 'Q': 2.5, 'L': 3.0, 'I': 3.5, 'O': 4.0
}

TRANSFER_TYPE_MIN_FLOORS = {
    'S': 5000, 'Q': 3000, 'L': 2000, 'I': 1500, 'O': 1000
}

def calculate_dynamic_threshold(user_avg_amount, user_std_amount, transfer_type='O', min_floor=None):
    transfer_type = str(transfer_type).upper()
    multiplier = TRANSFER_TYPE_MULTIPLIERS.get(transfer_type, 3.0)
    
    if min_floor is None:
        min_floor = TRANSFER_TYPE_MIN_FLOORS.get(transfer_type, 2000)
    
    statistical_limit = user_avg_amount + (multiplier * user_std_amount)
    final_limit = max(statistical_limit, min_floor)
    return final_limit

def calculate_all_limits(user_avg_amount, user_std_amount):
    limits = {}
    for transfer_type in ['S', 'I', 'L', 'Q', 'O']:
        limits[transfer_type] = calculate_dynamic_threshold(
            user_avg_amount, user_std_amount, transfer_type
        )
    return limits

def check_rule_violation(transaction_amount, user_avg_amount, user_std_amount, transfer_type='O', window_txn_count=0, ml_velocity_anomaly=False):
    """
    Check violation - ML model decides velocity anomaly, no hardcoded limit.
    
    Args:
        transaction_amount: Current transaction amount
        user_avg_amount: User's average transaction amount
        user_std_amount: User's std deviation of transactions
        transfer_type: Type of transfer (S, Q, L, I, O)
        window_txn_count: Number of transactions in time window (for display only)
        ml_velocity_anomaly: Whether ML model flagged velocity as anomaly
    """
    threshold = calculate_dynamic_threshold(user_avg_amount, user_std_amount, transfer_type)
    
    if ml_velocity_anomaly:
        reason = f"High transaction frequency detected ({window_txn_count} transactions in time window) - ML flagged as anomaly"
        return True, reason, threshold
    
    if transaction_amount > threshold:
        transfer_type_names = {
            'S': 'Overseas', 'I': 'Ajman', 'L': 'UAE', 'Q': 'Quick Remittance', 'O': 'Own Account'
        }
        type_name = transfer_type_names.get(str(transfer_type).upper(), transfer_type)
        reason = f"Transaction amount ({transaction_amount:.2f}) exceeds {type_name} threshold ({threshold:.2f})"
        return True, reason, threshold
    
    return False, "Transaction within normal limits", threshold

def get_user_stats(df, customer_id, customer_col='CustomerId'):
    user_data = df[df[customer_col] == customer_id]
    if len(user_data) == 0:
        return {'user_avg_amount': 0, 'user_std_amount': 0, 'user_max_amount': 0, 'user_txn_frequency': 0}
    
    amount_col = 'transaction_amount' if 'transaction_amount' in user_data.columns else 'Amount'
    if amount_col not in user_data.columns:
        return {'user_avg_amount': 0, 'user_std_amount': 0, 'user_max_amount': 0, 'user_txn_frequency': 0}
    
    return {
        'user_avg_amount': user_data[amount_col].mean(),
        'user_std_amount': user_data[amount_col].std() if len(user_data) > 1 else 0,
        'user_max_amount': user_data[amount_col].max(),
        'user_txn_frequency': len(user_data)
    }
