TRANSFER_MULTIPLIERS = {'S': 2.0, 'Q': 2.5, 'L': 3.0, 'I': 3.5, 'O': 4.0}
TRANSFER_MIN_FLOORS = {'S': 5000, 'Q': 3000, 'L': 2000, 'I': 1500, 'O': 1000}
# MAX_VELOCITY_LIMIT = 10
# BURST_LIMIT_30S = 5

def calculate_threshold(user_avg, user_std, transfer_type='O'):
    transfer_type = str(transfer_type).upper()
    multiplier = TRANSFER_MULTIPLIERS.get(transfer_type, 3.0)
    min_floor = TRANSFER_MIN_FLOORS.get(transfer_type, 2000)
    limit = user_avg + (multiplier * user_std)
    return max(limit, min_floor)

def calculate_all_limits(user_avg, user_std):
    return {t: calculate_threshold(user_avg, user_std, t) for t in ['S', 'I', 'L', 'Q', 'O']}

def check_rule_violation(amount, user_avg, user_std, transfer_type='O', spending_so_far=0):
    threshold = calculate_threshold(user_avg, user_std, transfer_type)
    total_spending = amount + spending_so_far

    if total_spending > threshold:
        return True, f"Amount exceeds allowed threshold ({threshold:.2f})", threshold

    return False, "Within limits", threshold
# def check_rule_violation(amount, user_avg, user_std, transfer_type='O', ml_anomaly=False, txn_count=0, spending_so_far=0, txn_count_30s=0):
    threshold = calculate_threshold(user_avg, user_std, transfer_type)
    type_names = {'S': 'Overseas', 'I': 'Ajman', 'L': 'UAE', 'Q': 'Quick Remittance', 'O': 'Own Account'}
    total_spending = amount + spending_so_far

    violation_reasons = []
    is_violated = False

    # if txn_count_30s >= BURST_LIMIT_30S:
    #     return True, f"Burst Limit Reached: {txn_count_30s} transactions", 0

    # if txn_count > MAX_VELOCITY_LIMIT:
    #     is_violated = True
    #     violation_reasons.append(f"Velocity Limit Exceeded ({txn_count} txns in 10 min) - Rule Blocked")

    if ml_anomaly:
        is_violated = True
        violation_reasons.append(f"ML Anomaly: Unusual burst detected ({txn_count} txns in short window)")
        
    if total_spending > threshold:
        is_violated = True
        name = type_names.get(str(transfer_type).upper(), transfer_type)
        violation_reasons.append(f"Total Spending ({total_spending:.2f}) exceeds {name} limit ({threshold:.2f}).")

    final_reason = " | ".join(violation_reasons) if violation_reasons else "Within normal limits"
    
    return is_violated, final_reason, threshold