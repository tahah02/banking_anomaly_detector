TRANSFER_MULTIPLIERS = {'S': 2.0, 'Q': 2.5, 'L': 3.0, 'I': 3.5, 'O': 4.0}
TRANSFER_MIN_FLOORS = {'S': 5000, 'Q': 3000, 'L': 2000, 'I': 1500, 'O': 1000}

def calculate_threshold(user_avg, user_std, transfer_type='O'):
    transfer_type = str(transfer_type).upper()
    multiplier = TRANSFER_MULTIPLIERS.get(transfer_type, 3.0)
    min_floor = TRANSFER_MIN_FLOORS.get(transfer_type, 2000)
    limit = user_avg + (multiplier * user_std)
    return max(limit, min_floor)

def calculate_all_limits(user_avg, user_std):
    return {t: calculate_threshold(user_avg, user_std, t) for t in ['S', 'I', 'L', 'Q', 'O']}

def check_rule_violation(amount, user_avg, user_std, transfer_type='O', ml_anomaly=False, txn_count=0):
    threshold = calculate_threshold(user_avg, user_std, transfer_type)
    type_names = {'S': 'Overseas', 'I': 'Ajman', 'L': 'UAE', 'Q': 'Quick Remittance', 'O': 'Own Account'}
    
    if ml_anomaly:
        return True, f"Burst detected ({txn_count} txns in 10 min) - ML flagged", threshold
    
    if amount > threshold:
        name = type_names.get(str(transfer_type).upper(), transfer_type)
        return True, f"Amount ({amount:.2f}) exceeds {name} limit ({threshold:.2f})", threshold
    
    return False, "Within normal limits", threshold
