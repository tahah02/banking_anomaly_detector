import streamlit as st
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict

st.set_page_config(page_title="Banking Fraud Detection", page_icon="🏦", layout="wide")

def init_state():
    defaults = {'logged_in': False, 'customer_id': None, 'result': None, 
                'history': defaultdict(list), 'session_txns': defaultdict(lambda: {'count': 0, 'amount': 0})}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_velocity(cid):
    now = datetime.now()
    hist = [t for t in st.session_state.history.get(str(cid), []) if (now - t).total_seconds() < 3600]
    st.session_state.history[str(cid)] = hist
    cnt_10 = sum(1 for t in hist if (now - t).total_seconds() < 600) + 1
    cnt_1h = len(hist) + 1
    time_last = (now - max(hist)).total_seconds() if hist else 3600
    return {'txn_count_10min': cnt_10, 'txn_count_1hour': cnt_1h, 'time_since_last_txn': time_last}

def record_txn(cid, amount):
    st.session_state.history[str(cid)].append(datetime.now())
    st.session_state.session_txns[str(cid)]['count'] += 1
    st.session_state.session_txns[str(cid)]['amount'] += amount

def run_pipeline():
    from backend.utils import get_clean_csv_path, get_feature_engineered_path, get_model_path
    if not os.path.exists(get_clean_csv_path()):
        st.warning("data/Clean.csv not found")
        return False
    if not os.path.exists(get_feature_engineered_path()):
        with st.spinner("Engineering features..."):
            from backend.feature_engineering import engineer_features
            engineer_features()
    if not os.path.exists(get_model_path()):
        with st.spinner("Training model..."):
            from backend.model_training import train_model
            train_model()
    return True

@st.cache_data
def load_data():
    from backend.utils import get_feature_engineered_path, get_clean_csv_path
    path = get_feature_engineered_path() if os.path.exists(get_feature_engineered_path()) else get_clean_csv_path()
    return pd.read_csv(path) if os.path.exists(path) else None

def login_page():
    st.title("🏦 Banking Fraud Detection System")
    st.subheader("Login")
    df = load_data()
    if df is None:
        if not run_pipeline(): return
        st.rerun()
    customers = sorted([str(c) for c in df['CustomerId'].dropna().unique()])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Select Customer ID")
        cid = st.selectbox("Customer ID", customers)
        pwd = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if pwd == "12345":
                st.session_state.logged_in, st.session_state.customer_id = True, cid
                st.rerun()
            else:
                st.error("Invalid password")
        st.info("Password: 12345")

def dashboard():
    from backend.hybrid_decision import make_decision
    from backend.rule_engine import calculate_all_limits
    from backend.model_training import load_model
    
    cid = st.session_state.customer_id
    df = load_data()
    amt_col = 'transaction_amount' if 'transaction_amount' in df.columns else 'Amount'
    cust_data = df[df['CustomerId'].astype(str) == str(cid)]
    
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Customer ID:** {cid}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in, st.session_state.customer_id, st.session_state.result = False, None, None
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Customer Statistics")
    if len(cust_data) > 0:
        avg = cust_data[amt_col].mean()
        max_amt = cust_data[amt_col].max()
        total_txns = len(cust_data)
        std = cust_data[amt_col].std() if len(cust_data) > 1 else 0
        
        st.sidebar.markdown("**Average Transaction:**")
        st.sidebar.markdown(f"AED {avg:,.2f}")
        st.sidebar.markdown("**Max Transaction:**")
        st.sidebar.markdown(f"AED {max_amt:,.2f}")
        st.sidebar.markdown("**Total Transactions:**")
        st.sidebar.markdown(f"{total_txns}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Transfer Type Limits")
        limits = calculate_all_limits(avg, std)
        st.sidebar.markdown(f"**O - Own Account:** AED {limits['O']:,.2f}")
        st.sidebar.markdown(f"**I - Ajman:** AED {limits['I']:,.2f}")
        st.sidebar.markdown(f"**L - UAE:** AED {limits['L']:,.2f}")
        st.sidebar.markdown(f"**Q - Quick (Medium Risk):** AED {limits['Q']:,.2f}")
        st.sidebar.markdown(f"**S - Overseas (High Risk):** AED {limits['S']:,.2f}")
    
    st.title("Fraud Detection Dashboard")
    
    st.subheader("Step 1: Select Account")
    st.markdown("**Select Account to Debit**")
    accounts = [str(a) for a in cust_data['FromAccountNo'].dropna().unique()] if 'FromAccountNo' in cust_data.columns and len(cust_data) > 0 else ["Default"]
    account = st.selectbox("", accounts, label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Step 2: Transaction Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Transaction Amount (AED)**")
        amount = st.number_input("", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0, label_visibility="collapsed")
    with c2:
        st.markdown("**Transfer Type**")
        types = {'O': 'O - Own Account Transfer', 'I': 'I - Within Ajman', 'L': 'L - Within UAE', 'Q': 'Q - Quick Remittance', 'S': 'S - Overseas Transaction'}
        t_type = st.selectbox("", list(types.keys()), format_func=lambda x: types[x], label_visibility="collapsed")
    with c3:
        st.markdown("**Bank Country**")
        countries = ['UAE', 'USA', 'UK', 'India', 'Pakistan', 'Philippines', 'Egypt', 'Other']
        country = st.selectbox("", countries, label_visibility="collapsed", key="country")
    
    st.markdown("---")
    st.subheader("Step 3: Process Transaction")
    
    if st.button("Process Transaction", type="primary", use_container_width=True):
        user_stats = {
            'user_avg_amount': cust_data[amt_col].mean() if len(cust_data) > 0 else 5000,
            'user_std_amount': cust_data[amt_col].std() if len(cust_data) > 1 else 2000,
            'user_max_amount': cust_data[amt_col].max() if len(cust_data) > 0 else 15000,
            'user_txn_frequency': len(cust_data)
        }
        vel = get_velocity(cid)
        txn = {'amount': amount, 'transfer_type': t_type, 'bank_country': country, **vel}
        try:
            model, feats = load_model()
            st.session_state.result = make_decision(txn, user_stats, model, feats)
        except:
            st.session_state.result = make_decision(txn, user_stats)
        st.session_state.result['amount'] = amount
    
    st.markdown("---")
    st.subheader("Step 4: Transaction Result")
    
    if st.session_state.result:
        r = st.session_state.result
        if r['is_fraud']:
            st.error("FRAUD ALERT - Transaction Flagged!")
            st.markdown("### Reasons for Flag:")
            for reason in r['reasons']:
                st.warning(f"- {reason}")
            st.markdown(f"**Dynamic Threshold:** AED {r['threshold']:.2f}")
            st.markdown(f"**Risk Score:** {r['risk_score']:.4f}")
            
            st.markdown("---")
            st.markdown("### Action Required")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve Transaction", type="primary", use_container_width=True):
                    record_txn(cid, r['amount'])
                    st.success("Transaction APPROVED by user!")
                    st.balloons()
                    st.session_state.result = None
                    st.rerun()
            with c2:
                if st.button("Reject Transaction", type="secondary", use_container_width=True):
                    st.error("Transaction REJECTED!")
                    st.session_state.result = None
                    st.rerun()
        else:
            st.success("SAFE TRANSACTION - No Fraud Detected!")
            st.markdown(f"**Dynamic Threshold:** AED {r['threshold']:.2f}")
            st.markdown("Transaction is within normal parameters.")
            if st.button("Process Another Transaction", use_container_width=True):
                record_txn(cid, r['amount'])
                st.session_state.result = None
                st.rerun()

def main():
    init_state()
    if not os.path.exists('data/Clean.csv'):
        st.title("🏦 Banking Fraud Detection")
        st.warning("Please ensure data/Clean.csv exists")
        return
    if not run_pipeline(): return
    if st.session_state.logged_in:
        dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()
