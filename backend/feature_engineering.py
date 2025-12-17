import pandas as pd
import numpy as np
from backend.utils import ensure_data_dir, get_clean_csv_path, get_feature_engineered_path, TRANSFER_TYPE_ENCODED, TRANSFER_TYPE_RISK

def engineer_features():
    ensure_data_dir()
    df = pd.read_csv(get_clean_csv_path())
    
    if 'CreateDate' in df.columns:
        df['CreateDate'] = pd.to_datetime(df['CreateDate'], errors='coerce')
    
    df['transaction_amount'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0)
    
    if 'TransferType' in df.columns:
        df['flag_amount'] = df['TransferType'].apply(lambda x: 1 if str(x).upper() == 'S' else 0)
        df['transfer_type_encoded'] = df['TransferType'].apply(lambda x: TRANSFER_TYPE_ENCODED.get(str(x).upper(), 0))
        df['transfer_type_risk'] = df['TransferType'].apply(lambda x: TRANSFER_TYPE_RISK.get(str(x).upper(), 0.5))
    else:
        df['flag_amount'], df['transfer_type_encoded'], df['transfer_type_risk'] = 0, 0, 0.5
    
    df['channel_encoded'] = 0
    if 'ChannelId' in df.columns:
        mapping = {v: i for i, v in enumerate(df['ChannelId'].dropna().unique())}
        df['channel_encoded'] = df['ChannelId'].map(mapping).fillna(0).astype(int)
    
    if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
        df['hour'] = df['CreateDate'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['CreateDate'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    else:
        df['hour'], df['day_of_week'], df['is_weekend'], df['is_night'] = 12, 0, 0, 0
    
    if 'CustomerId' in df.columns:
        stats = df.groupby('CustomerId')['transaction_amount'].agg(['mean', 'std', 'max', 'count'])
        stats.columns = ['user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency']
        stats['user_std_amount'] = stats['user_std_amount'].fillna(0)
        df = df.merge(stats.reset_index(), on='CustomerId', how='left')
        df['deviation_from_avg'] = abs(df['transaction_amount'] - df['user_avg_amount'])
        df['amount_to_max_ratio'] = df['transaction_amount'] / df['user_max_amount'].replace(0, 1)
        
        if 'TransferType' in df.columns:
            df['intl_ratio'] = df.groupby('CustomerId')['flag_amount'].transform('mean')
        else:
            df['intl_ratio'] = 0
        
        if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
            df = df.sort_values(['CustomerId', 'CreateDate'])
            df.set_index('CreateDate', inplace=True)
            df['txn_count_30s'] = df.groupby('CustomerId')['transaction_amount']\
                                                .rolling('30s').count().values
                        
                        # Calculate 10-minute density (The "Velocity" feature)
            df['txn_count_10min'] = df.groupby('CustomerId')['transaction_amount']\
                                                .rolling('10min').count().values
                        
            df.reset_index(inplace=True)
                        
                        # Calculate time gap between transactions
            df['time_since_last'] = df.groupby('CustomerId')['CreateDate'].diff().dt.total_seconds().fillna(3600)
        else:
            df['txn_count_30s'], df['txn_count_10min'], df['time_since_last'] = 1, 1, 3600
            df['month_period'] = df['CreateDate'].dt.to_period('M')
            df['current_month_spending'] = df.groupby(['CustomerId', 'month_period'])['transaction_amount']\
                                             .transform(lambda x: x.cumsum().shift(1)).fillna(0)
            df['time_since_last'] = df.groupby('CustomerId')['CreateDate'].diff().dt.total_seconds().fillna(3600)
            df['recent_burst'] = (df['time_since_last'] < 300).astype(int)               
    else:
        df['user_avg_amount'] = df['transaction_amount'].mean()
        df['user_std_amount'] = df['transaction_amount'].std()
        df['user_max_amount'] = df['transaction_amount'].max()
        df['user_txn_frequency'] = len(df)
        df['deviation_from_avg'], df['amount_to_max_ratio'] = 0, 0
        df['intl_ratio'], df['time_since_last'], df['recent_burst'] = 0, 3600, 0
        df['current_month_spending'] = 0
        df['time_since_last'], df['recent_burst'] = 3600, 0
    
    df['txn_count_10min'], df['txn_count_1hour'] = 1, 1
    df['rolling_std'] = df.groupby('CustomerId')['transaction_amount'].transform(
        lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std()
    ).fillna(0) if 'CustomerId' in df.columns else 0
    
    df.to_csv(get_feature_engineered_path(), index=False)
    print(f"Features saved: {df.shape}")
    return df

if __name__ == "__main__":
    engineer_features()
