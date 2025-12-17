import streamlit as st
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import csv
import os

def save_transaction_to_csv(cid, amount, t_type, status="Approved"):
    file_name = 'transaction_history.csv'
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['CustomerID', 'Amount', 'Type', 'Status', 'Timestamp'])
        writer.writerow([cid, amount, t_type, status, datetime.now()])

st.set_page_config(page_title="Banking Fraud Detection", page_icon="🏦", layout="wide")

def init_state():
    defaults = {'logged_in': False, 'customer_id': None, 'result': None, 
                'history': defaultdict(list), 'session_txns': defaultdict(lambda: {'count': 0, 'amount': 0})}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_velocity(cid):
    now = datetime.now()
    cid = str(cid) 
    if 'user_spending_history' not in st.session_state:
        st.session_state.user_spending_history = {}
        
    hist = st.session_state.history.get(cid, [])
    recent_txns = [t for t in hist if (now - t).total_seconds() < 600]

    cnt_10 = len(recent_txns) + 1  
    cnt_1h = len(st.session_state.history.get(cid, [])) + 1 

    if recent_txns:

        first_txn_in_burst = min(recent_txns)
        diff_seconds = (now - first_txn_in_burst).total_seconds()
        
        if diff_seconds < 60:
            real_time_window = f"{diff_seconds:.2f} sec"
        else:
            minutes = int(diff_seconds / 60) + 1
            real_time_window = f"{minutes} min"
    else:
        real_time_window = "0.00 sec"


    time_last = (now - max(hist)).total_seconds() if hist else 3600
    

    return {
        'txn_count_10min': cnt_10, 
        'txn_count_1hour': cnt_1h, 
        'time_since_last_txn': time_last,
        'real_time_window': real_time_window 
    }

def record_txn(cid, amount):
    cid = str(cid) 
    if cid not in st.session_state.session_txns:
        st.session_state.session_txns[cid] = {'count': 0, 'amount': 0}

    st.session_state.history[cid].append(datetime.now())

    st.session_state.session_txns[cid]['count'] += 1
    st.session_state.session_txns[cid]['amount'] += amount

    st.session_state['session_txns'] = st.session_state['session_txns']

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
    if 'user_spending_history' not in st.session_state:
        st.session_state.user_spending_history = {}
    if 'history' not in st.session_state:
        st.session_state.history = {}
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'session_txns' not in st.session_state:
        st.session_state.session_txns = {}

    cid = str(st.session_state.customer_id) 
    df = load_data()
    amt_col = 'transaction_amount' if 'transaction_amount' in df.columns else 'Amount'
    cust_data = df[df['CustomerId'].astype(str) == str(cid)]

    if cid not in st.session_state.session_txns:
        st.session_state.session_txns[cid] = {'count': 0, 'amount': 0}
    
    current_session_count = st.session_state.session_txns[cid]['count']
    csv_count = len(cust_data)
    total_txns = csv_count + current_session_count

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
        std = cust_data[amt_col].std() if len(cust_data) > 1 else 0
        
        st.sidebar.markdown("**Average Transaction:**")
        st.sidebar.markdown(f"AED {avg:,.2f}")
        st.sidebar.markdown("**Max Transaction:**")
        st.sidebar.markdown(f"AED {max_amt:,.2f}")
        
        st.sidebar.metric("Total Transactions", total_txns, delta=f"+{current_session_count} New" if current_session_count > 0 else None)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Transfer Type Limits")
        limits = calculate_all_limits(avg, std)
        st.sidebar.markdown(f"**O - Own Account:** AED {limits['O']:,.2f}")
        st.sidebar.markdown(f"**I - Ajman:** AED {limits['I']:,.2f}")
        st.sidebar.markdown(f"**L - UAE:** AED {limits['L']:,.2f}")
        st.sidebar.markdown(f"**Q - Quick:** AED {limits['Q']:,.2f}")
        st.sidebar.markdown(f"**S - Overseas:** AED {limits['S']:,.2f}")

    st.title("Fraud Detection Dashboard")
    
    st.subheader("Step 1: Select Account")
    accounts = [str(a) for a in cust_data['FromAccountNo'].dropna().unique()] if 'FromAccountNo' in cust_data.columns and len(cust_data) > 0 else ["Default"]
    account = st.selectbox("Select Account", accounts, label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Step 2: Transaction Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Transaction Amount (AED)**")
        amount = st.number_input("", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0, key="amt_input")
    with c2:
        st.markdown("**Transfer Type**")
        types = {'O': 'O - Own Account', 'I': 'I - Within Ajman', 'L': 'L - Within UAE', 'Q': 'Q - Quick Remittance', 'S': 'S - Overseas'}
        t_type = st.selectbox("", list(types.keys()), format_func=lambda x: types[x], key="type_input")
    with c3:
        st.markdown("**Bank Country**")
        countries = ['UAE', 'USA', 'UK', 'India', 'Pakistan', 'Philippines', 'Egypt', 'Other']
        country = st.selectbox("", countries, key="country_input")
    
    st.markdown("---")
    st.subheader("Step 3: Process Transaction")
    if st.button("Process Transaction", type="primary", use_container_width=True):

        csv_spending = cust_data[amt_col].sum() if len(cust_data) > 0 else 0
        live_spending = st.session_state.user_spending_history.get(str(cid), 0.0)
        current_spending = csv_spending + live_spending

        current_type_txns = cust_data[cust_data['TransferType'] == t_type]
        
        if len(current_type_txns) > 0:
            specific_avg = current_type_txns[amt_col].mean()
            specific_std = current_type_txns[amt_col].std() if len(current_type_txns) > 1 else 0
            specific_max = current_type_txns[amt_col].max()
        else:
            specific_avg = cust_data[amt_col].mean() if len(cust_data) > 0 else 5000
            specific_std = cust_data[amt_col].std() if len(cust_data) > 1 else 2000
            specific_max = cust_data[amt_col].max() if len(cust_data) > 0 else 15000

        total_txns_count = len(cust_data)
        intl_ratio = 0.0
        high_risk_ratio = 0.0
        if total_txns_count > 0:
            count_s = len(cust_data[cust_data['TransferType'] == 'S'])
            intl_ratio = count_s / total_txns_count
            count_q = len(cust_data[cust_data['TransferType'] == 'Q'])
            high_risk_ratio = (count_s + count_q) / total_txns_count

        user_stats = {
            'user_avg_amount': specific_avg,  
            'user_std_amount': specific_std,
            'user_max_amount': specific_max,
            'user_txn_frequency': len(cust_data), 
            'user_international_ratio': intl_ratio,
            'user_high_risk_txn_ratio': high_risk_ratio,
            'current_month_spending': current_spending 
        }

        vel = get_velocity(cid)
        txn = {'amount': amount, 'transfer_type': t_type, 'bank_country': country, **vel}
        
        try:
            model, feats = load_model()
            st.session_state.result = make_decision(txn, user_stats, model, feats)
        except:
            st.session_state.result = make_decision(txn, user_stats)
            
        st.session_state.result['amount'] = amount
        st.rerun() 

    st.markdown("---")
    st.subheader("Step 4: Transaction Result")

    if st.session_state.result:
        r = st.session_state.result
        if r['is_fraud']:
            st.error("FRAUD ALERT - Transaction Flagged!")
            for reason in r['reasons']:
                st.warning(f"- {reason}")
            
            st.markdown(f"**Risk Score:** {r['risk_score']:.4f}")
            
            c1, c2 = st.columns(2)
            with c1:

                if st.button("Approve Transaction (Force)", type="primary"):

                    save_transaction_to_csv(cid, r['amount'], t_type, "Force Approved")

                    cid_str = str(cid)
                    curr = st.session_state.user_spending_history.get(cid_str, 0.0)
                    st.session_state.user_spending_history[cid_str] = curr + r['amount']
                    st.session_state.session_txns[cid]['count'] += 1
                    
                    st.success("Approved & Saved!")
                    st.session_state.result = None
                    st.rerun()
                    
            with c2:

                if st.button("Reject Transaction"):
                    st.error("Rejected!")
                    st.session_state.result = None
                    st.rerun()

        else:
            st.success("SAFE TRANSACTION ✅")
            st.info(f"Amount: {r['amount']} AED | Threshold: {r['threshold']:.2f}")

            if st.button("Process Another Transaction", type="primary"):
                save_transaction_to_csv(cid, r['amount'], t_type, "Approved")

                cid_str = str(cid)
                curr = st.session_state.user_spending_history.get(cid_str, 0.0)
                st.session_state.user_spending_history[cid_str] = curr + r['amount']
                st.session_state.session_txns[cid]['count'] += 1
                
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
