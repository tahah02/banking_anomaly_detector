import pandas as pd
import numpy as np
from backend.utils import (
    ensure_data_dir, get_clean_csv_path, get_feature_engineered_path,
    MERCHANT_TYPE_ENCODED, MERCHANT_RISK_SCORES
)

def engineer_features():
    """
    Create all 26+ features for fraud detection.
    Includes transaction velocity features for ML-based frequency detection.
    """
    ensure_data_dir()
    
    clean_path = get_clean_csv_path()
    feature_path = get_feature_engineered_path()
    
    print(f"Loading clean data from: {clean_path}")
    df = pd.read_csv(clean_path)
    
    if 'CreateDate' in df.columns:
        df['CreateDate'] = pd.to_datetime(df['CreateDate'], errors='coerce')
    
    amount_col = None
    for col in ['Amount', 'TransactionAmount', 'amount']:
        if col in df.columns:
            amount_col = col
            break
    
    if amount_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            amount_col = numeric_cols[0]
    
    if amount_col:
        df['transaction_amount'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    else:
        df['transaction_amount'] = 0
    
    transfer_type_col = None
    for col in ['TransferType', 'MerchantType', 'Type']:
        if col in df.columns:
            transfer_type_col = col
            break
    
    if transfer_type_col:
        df['flag_amount'] = df[transfer_type_col].apply(lambda x: 1 if str(x).upper() == 'S' else 0)
        df['merchant_type_encoded'] = df[transfer_type_col].apply(
            lambda x: MERCHANT_TYPE_ENCODED.get(str(x).upper(), 0)
        )
        df['merchant_risk_score'] = df[transfer_type_col].apply(
            lambda x: MERCHANT_RISK_SCORES.get(str(x).upper(), 0.5)
        )
    else:
        df['flag_amount'] = 0
        df['merchant_type_encoded'] = 0
        df['merchant_risk_score'] = 0.5
    
    channel_col = None
    for col in ['ChannelId', 'Channel', 'TransactionChannel']:
        if col in df.columns:
            channel_col = col
            break
    
    if channel_col:
        channel_mapping = {val: idx for idx, val in enumerate(df[channel_col].dropna().unique())}
        df['channel_encoded'] = df[channel_col].map(channel_mapping).fillna(0).astype(int)
    else:
        df['channel_encoded'] = 0
    
    if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
        df['hour'] = df['CreateDate'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['CreateDate'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    else:
        df['hour'] = 12
        df['day_of_week'] = 0
        df['is_weekend'] = 0
        df['is_night'] = 0
    
    customer_col = None
    for col in ['CustomerId', 'CustomerID', 'customer_id', 'CUSTOMERID']:
        if col in df.columns:
            customer_col = col
            break
    
    if customer_col:
        user_stats = df.groupby(customer_col)['transaction_amount'].agg(['mean', 'std', 'max', 'count'])
        user_stats.columns = ['user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency']
        user_stats = user_stats.reset_index()
        user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)
        
        df = df.merge(user_stats, on=customer_col, how='left')
        
        df['deviation_from_user_avg'] = abs(df['transaction_amount'] - df['user_avg_amount'])
        df['amount_to_user_max_ratio'] = df['transaction_amount'] / df['user_max_amount'].replace(0, 1)
        
        if transfer_type_col:
            intl_ratio = df.groupby(customer_col).apply(
                lambda x: (x[transfer_type_col].str.upper() == 'S').sum() / len(x) if len(x) > 0 else 0
            ).reset_index()
            intl_ratio.columns = [customer_col, 'user_international_ratio']
            df = df.merge(intl_ratio, on=customer_col, how='left')
            
            high_risk_ratio = df.groupby(customer_col).apply(
                lambda x: (x[transfer_type_col].str.upper().isin(['S', 'Q'])).sum() / len(x) if len(x) > 0 else 0
            ).reset_index()
            high_risk_ratio.columns = [customer_col, 'user_high_risk_txn_ratio']
            df = df.merge(high_risk_ratio, on=customer_col, how='left')
        else:
            df['user_international_ratio'] = 0
            df['user_high_risk_txn_ratio'] = 0
        
        account_col = None
        for col in ['FromAccountNo', 'AccountNo', 'AccountNumber']:
            if col in df.columns:
                account_col = col
                break
        
        if account_col:
            account_counts = df.groupby(customer_col)[account_col].nunique().reset_index()
            account_counts.columns = [customer_col, 'num_accounts']
            df = df.merge(account_counts, on=customer_col, how='left')
            df['user_multiple_accounts_flag'] = (df['num_accounts'] > 1).astype(int)
            df.drop(columns=['num_accounts'], inplace=True)
            
            if 'BenId' in df.columns:
                cross_transfer = df.groupby(customer_col).apply(
                    lambda x: (x['BenId'] == -1).sum() / len(x) if len(x) > 0 else 0
                ).reset_index()
                cross_transfer.columns = [customer_col, 'cross_account_transfer_ratio']
                df = df.merge(cross_transfer, on=customer_col, how='left')
            else:
                df['cross_account_transfer_ratio'] = 0
        else:
            df['user_multiple_accounts_flag'] = 0
            df['cross_account_transfer_ratio'] = 0
        
        if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
            df_sorted = df.sort_values([customer_col, 'CreateDate'])
            df_sorted['time_diff_seconds'] = df_sorted.groupby(customer_col)['CreateDate'].diff().dt.total_seconds().fillna(3600)
            
            df['time_since_last_txn'] = df_sorted['time_diff_seconds']
            
            def count_txns_in_window(group, window_seconds=600):
                group = group.sort_values('CreateDate')
                counts = []
                for idx, row in group.iterrows():
                    current_time = row['CreateDate']
                    window_start = current_time - pd.Timedelta(seconds=window_seconds)
                    count = ((group['CreateDate'] >= window_start) & (group['CreateDate'] <= current_time)).sum()
                    counts.append(count)
                return pd.Series(counts, index=group.index)
            
            df['txn_count_10min'] = df.groupby(customer_col, group_keys=False).apply(
                lambda x: count_txns_in_window(x, 600)
            ).fillna(1)
            
            df['txn_count_1hour'] = df.groupby(customer_col, group_keys=False).apply(
                lambda x: count_txns_in_window(x, 3600)
            ).fillna(1)
            
            df['recent_activity_burst'] = (df_sorted['time_diff_seconds'] < 300).astype(int)
        else:
            df['time_since_last_txn'] = 3600
            df['txn_count_10min'] = 1
            df['txn_count_1hour'] = 1
            df['recent_activity_burst'] = 0
            
    else:
        df['user_avg_amount'] = df['transaction_amount'].mean()
        df['user_std_amount'] = df['transaction_amount'].std()
        df['user_max_amount'] = df['transaction_amount'].max()
        df['user_txn_frequency'] = len(df)
        df['deviation_from_user_avg'] = 0
        df['amount_to_user_max_ratio'] = 0
        df['user_international_ratio'] = 0
        df['user_high_risk_txn_ratio'] = 0
        df['user_multiple_accounts_flag'] = 0
        df['cross_account_transfer_ratio'] = 0
        df['time_since_last_txn'] = 3600
        df['txn_count_10min'] = 1
        df['txn_count_1hour'] = 1
        df['recent_activity_burst'] = 0
    
    df['rolling_std_amount'] = df.groupby(customer_col if customer_col else df.index)['transaction_amount'].transform(
        lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std()
    ).fillna(0)
    
    df['transaction_velocity'] = df.groupby(customer_col if customer_col else df.index)['transaction_amount'].transform('count')
    
    if 'BenId' in df.columns:
        ben_first_txn = df.groupby('BenId')['CreateDate'].transform('min') if 'CreateDate' in df.columns else None
        if ben_first_txn is not None:
            df['is_new_beneficiary'] = (df['CreateDate'] == ben_first_txn).astype(int)
        else:
            df['is_new_beneficiary'] = 0
        
        ben_counts = df.groupby('BenId').size().reset_index(name='beneficiary_txn_count_30d')
        df = df.merge(ben_counts, on='BenId', how='left')
        df['beneficiary_txn_count_30d'] = df['beneficiary_txn_count_30d'].fillna(0)
        
        ben_avg = df.groupby('BenId')['transaction_amount'].mean().reset_index()
        ben_avg.columns = ['BenId', 'ben_avg_amount']
        df = df.merge(ben_avg, on='BenId', how='left')
        df['beneficiary_risk_score'] = (df['transaction_amount'] / df['ben_avg_amount'].replace(0, 1)).clip(0, 1)
        df.drop(columns=['ben_avg_amount'], inplace=True)
    else:
        df['is_new_beneficiary'] = 0
        df['beneficiary_txn_count_30d'] = 0
        df['beneficiary_risk_score'] = 0.5
    
    country_col = None
    for col in ['BankCountry', 'Country', 'ToCountry']:
        if col in df.columns:
            country_col = col
            break
    
    if country_col:
        local_countries = ['UAE', 'AE', 'United Arab Emirates', 'AJMAN']
        df['geo_anomaly_flag'] = df[country_col].apply(
            lambda x: 0 if str(x).upper() in [c.upper() for c in local_countries] else 1
        )
    else:
        df['geo_anomaly_flag'] = df['flag_amount']
    
    df.to_csv(feature_path, index=False)
    print(f"Feature engineered data saved to: {feature_path}")
    print(f"Final shape: {df.shape}")
    
    feature_cols = [
        'transaction_amount', 'flag_amount', 'merchant_type_encoded', 'merchant_risk_score',
        'channel_encoded', 'deviation_from_user_avg', 'amount_to_user_max_ratio', 'rolling_std_amount',
        'transaction_velocity', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_international_ratio',
        'user_high_risk_txn_ratio', 'user_txn_frequency', 'user_multiple_accounts_flag',
        'cross_account_transfer_ratio', 'is_new_beneficiary', 'beneficiary_txn_count_30d',
        'beneficiary_risk_score', 'geo_anomaly_flag', 'recent_activity_burst',
        'time_since_last_txn', 'txn_count_10min', 'txn_count_1hour'
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Available features ({len(available_features)}): {available_features}")
    
    return df

if __name__ == "__main__":
    engineer_features()
