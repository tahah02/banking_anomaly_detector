import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

st.set_page_config(
    page_title="Banking Fraud Detection System",
    page_icon="",
    layout="wide"
)

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'customer_id' not in st.session_state:
        st.session_state.customer_id = None
    if 'transaction_result' not in st.session_state:
        st.session_state.transaction_result = None
    if 'transaction_history' not in st.session_state:
        st.session_state.transaction_history = defaultdict(list)

def get_transaction_counts(customer_id, window_minutes=10):
    """
    Get transaction counts in time window for a customer.
    Returns counts for different time windows.
    """
    now = datetime.now()
    history = st.session_state.transaction_history.get(str(customer_id), [])
    
    history = [t for t in history if (now - t).total_seconds() < 3600]
    st.session_state.transaction_history[str(customer_id)] = history
    
    window_10min = sum(1 for t in history if (now - t).total_seconds() < 600)
    window_1hour = len(history)
    
    if history:
        last_txn_time = max(history)
        time_since_last = (now - last_txn_time).total_seconds()
    else:
        time_since_last = 3600
    
    return {
        'txn_count_10min': window_10min + 1,
        'txn_count_1hour': window_1hour + 1,
        'time_since_last_txn': time_since_last
    }

def record_transaction(customer_id):
    """Record a transaction timestamp for a customer."""
    st.session_state.transaction_history[str(customer_id)].append(datetime.now())

def run_data_pipeline():
    """Run the complete data pipeline if files don't exist"""
    from backend.utils import get_clean_csv_path, get_feature_engineered_path, get_model_path
    
    if not os.path.exists(get_clean_csv_path()):
        st.warning("Clean data file not found. Please ensure data/Clean.csv exists.")
        return False
    
    if not os.path.exists(get_feature_engineered_path()):
        with st.spinner("Engineering features..."):
            from backend.feature_engineering import engineer_features
            engineer_features()
    
    if not os.path.exists(get_model_path()):
        with st.spinner("Training ML model..."):
            from backend.model_training import train_model
            train_model()
        st.success("Model training completed!")
    
    return True

@st.cache_data
def load_clean_data():
    """Load clean data for customer selection"""
    from backend.utils import get_clean_csv_path, get_feature_engineered_path
    
    feature_path = get_feature_engineered_path()
    clean_path = get_clean_csv_path()
    
    if os.path.exists(feature_path):
        return pd.read_csv(feature_path)
    elif os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    return None

def get_customer_col(df):
    """Find customer ID column"""
    for col in ['CustomerId', 'CustomerID', 'customer_id', 'CUSTOMERID']:
        if col in df.columns:
            return col
    return None

def get_account_col(df):
    """Find account number column"""
    for col in ['FromAccountNo', 'AccountNo', 'AccountNumber', 'Account']:
        if col in df.columns:
            return col
    return None

def get_amount_col(df):
    """Find amount column"""
    for col in ['transaction_amount', 'Amount', 'TransactionAmount', 'amount']:
        if col in df.columns:
            return col
    return None

def login_page():
    """Display login page"""
    st.title("Banking Fraud Detection System")
    st.subheader("Login")
    
    df = load_clean_data()
    
    if df is None:
        st.warning("Data not loaded. Please ensure data files exist.")
        if not run_data_pipeline():
            return
        st.rerun()
        return
    
    customer_col = get_customer_col(df)
    
    if customer_col is None:
        st.error("Customer ID column not found in data")
        return
    
    valid_customers = df[customer_col].dropna().unique().tolist()
    valid_customers = [str(c) for c in valid_customers]
    valid_customers.sort()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Select Your Customer ID")
        selected_customer = st.selectbox(
            "Customer ID",
            options=valid_customers,
            key="login_customer_select"
        )
        
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if password == "12345":
                st.session_state.logged_in = True
                st.session_state.customer_id = selected_customer
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid password. Please try again.")
        
        st.markdown("---")
        st.info("Password for all users: 12345")

def main_dashboard():
    """Display main fraud detection dashboard"""
    from backend.hybrid_decision import make_hybrid_decision
    from backend.rule_engine import get_user_stats, calculate_all_limits
    from backend.model_training import load_model
    from backend.utils import MERCHANT_TYPE_MAPPING
    
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Customer ID:** {st.session_state.customer_id}")
    
    velocity_info = get_transaction_counts(st.session_state.customer_id)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Activity")
    st.sidebar.metric("Transactions (10 min)", velocity_info['txn_count_10min'] - 1)
    st.sidebar.metric("Transactions (1 hour)", velocity_info['txn_count_1hour'] - 1)
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.customer_id = None
        st.session_state.transaction_result = None
        st.rerun()
    
    st.title("Fraud Detection Dashboard")
    
    df = load_clean_data()
    
    if df is None:
        st.error("Data not loaded")
        return
    
    customer_col = get_customer_col(df)
    account_col = get_account_col(df)
    amount_col = get_amount_col(df)
    
    customer_id = st.session_state.customer_id
    
    try:
        if df[customer_col].dtype != object:
            customer_data = df[df[customer_col] == int(customer_id)]
        else:
            customer_data = df[df[customer_col].astype(str) == str(customer_id)]
    except:
        customer_data = df[df[customer_col].astype(str) == str(customer_id)]
    
    st.subheader("Step 1: Select Account")
    
    if account_col and len(customer_data) > 0:
        accounts = customer_data[account_col].dropna().unique().tolist()
        accounts = [str(a) for a in accounts]
        
        if len(accounts) == 0:
            st.warning("No accounts found for this customer")
            accounts = ["Default Account"]
        
        selected_account = st.selectbox(
            "Select Account to Debit",
            options=accounts,
            key="account_select"
        )
    else:
        selected_account = "Default Account"
        st.info("Account information not available")
    
    st.markdown("---")
    st.subheader("Step 2: Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input(
            "Transaction Amount (AED)",
            min_value=0.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
            key="amount_input"
        )
    
    with col2:
        transfer_types = {
            'O': 'Own Account Transfer',
            'I': 'Within Ajman',
            'L': 'Within UAE',
            'Q': 'Quick Remittance',
            'S': 'Overseas Transaction'
        }
        transfer_type = st.selectbox(
            "Transfer Type",
            options=list(transfer_types.keys()),
            format_func=lambda x: f"{x} - {transfer_types[x]}",
            key="transfer_type_select"
        )
    
    with col3:
        bank_countries = ['UAE', 'USA', 'UK', 'India', 'Pakistan', 'Philippines', 'Egypt', 'Other']
        bank_country = st.selectbox(
            "Bank Country",
            options=bank_countries,
            key="country_select"
        )
    
    st.markdown("---")
    st.subheader("Step 3: Analyze Transaction")
    
    if st.button("Process Transaction", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction..."):
            if amount_col and len(customer_data) > 0:
                user_stats = {
                    'user_avg_amount': customer_data[amount_col].mean() if amount_col in customer_data.columns else 0,
                    'user_std_amount': customer_data[amount_col].std() if amount_col in customer_data.columns and len(customer_data) > 1 else 0,
                    'user_max_amount': customer_data[amount_col].max() if amount_col in customer_data.columns else 0,
                    'user_txn_frequency': len(customer_data)
                }
            else:
                user_stats = {
                    'user_avg_amount': 5000,
                    'user_std_amount': 2000,
                    'user_max_amount': 15000,
                    'user_txn_frequency': 10
                }
            
            velocity_info = get_transaction_counts(customer_id)
            
            transaction_data = {
                'amount': amount,
                'transfer_type': transfer_type,
                'bank_country': bank_country,
                'account': selected_account,
                'txn_count_10min': velocity_info['txn_count_10min'],
                'txn_count_1hour': velocity_info['txn_count_1hour'],
                'time_since_last_txn': velocity_info['time_since_last_txn']
            }
            
            try:
                model, features = load_model()
                result = make_hybrid_decision(transaction_data, user_stats, model, features)
            except Exception as e:
                result = make_hybrid_decision(transaction_data, user_stats)
            
            st.session_state.transaction_result = result
    
    st.markdown("---")
    st.subheader("Step 4: Transaction Result")
    
    if st.session_state.transaction_result:
        result = st.session_state.transaction_result
        
        if result['is_fraud']:
            st.error("FRAUD ALERT - Transaction Flagged!")
            
            st.markdown("### Reasons for Flag:")
            for reason in result['reasons']:
                st.warning(f"- {reason}")
            
            st.markdown(f"**Dynamic Threshold:** AED {result['threshold']:.2f}")
            st.markdown(f"**Risk Score:** {result['risk_score']:.4f}")
            
            if result.get('velocity_anomaly'):
                st.info("This transaction was flagged due to high transaction frequency detected by ML model.")
            
            st.markdown("---")
            st.markdown("### Action Required")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Approve Transaction", type="primary", use_container_width=True):
                    record_transaction(customer_id)
                    st.success("Transaction APPROVED by user!")
                    st.balloons()
                    st.session_state.transaction_result = None
                    st.rerun()
            
            with col2:
                if st.button("Reject Transaction", type="secondary", use_container_width=True):
                    st.error("Transaction REJECTED!")
                    st.session_state.transaction_result = None
                    st.rerun()
        else:
            st.success("SAFE TRANSACTION - No Fraud Detected!")
            
            st.markdown(f"**Dynamic Threshold:** AED {result['threshold']:.2f}")
            st.markdown("Transaction is within normal parameters.")
            
            if st.button("Confirm & Process Transaction", type="primary", use_container_width=True):
                record_transaction(customer_id)
                st.success("Transaction processed successfully!")
                st.session_state.transaction_result = None
                st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Customer Statistics")
    
    if amount_col and len(customer_data) > 0:
        avg_amount = customer_data[amount_col].mean()
        std_amount = customer_data[amount_col].std() if len(customer_data) > 1 else 0
        max_amount = customer_data[amount_col].max()
        txn_count = len(customer_data)
        
        st.sidebar.metric("Average Transaction", f"AED {avg_amount:,.2f}")
        st.sidebar.metric("Max Transaction", f"AED {max_amount:,.2f}")
        st.sidebar.metric("Total Transactions", txn_count)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Transfer Type Limits")
        
        limits = calculate_all_limits(avg_amount, std_amount)
        
        transfer_type_names = {
            'S': 'Overseas',
            'I': 'Ajman',
            'L': 'UAE',
            'Q': 'Quick',
            'O': 'Own Account'
        }
        
        for t_type in ['O', 'I', 'L', 'Q', 'S']:
            limit = limits.get(t_type, 0)
            name = transfer_type_names.get(t_type, t_type)
            risk_label = ""
            if t_type == 'S':
                risk_label = " (High Risk)"
            elif t_type == 'Q':
                risk_label = " (Medium Risk)"
            st.sidebar.markdown(f"**{t_type} - {name}{risk_label}:** AED {limit:,.2f}")
    else:
        st.sidebar.info("No historical data available")

def main():
    init_session_state()
    
    if not os.path.exists('data/Clean.csv'):
        st.title("Banking Fraud Detection System")
        st.warning("Data file not found. Please upload or create data/Clean.csv")
        return
    
    if not run_data_pipeline():
        return
    
    if st.session_state.logged_in:
        main_dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()
