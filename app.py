import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from collections import defaultdict
import csv

# ---------------------- UTILITY FUNCTIONS ----------------------

def save_transaction_to_csv(cid, amount, t_type, status="Approved"):
    file_name = 'transaction_history.csv'
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['CustomerID', 'Amount', 'Type', 'Status', 'Timestamp'])
        writer.writerow([cid, amount, t_type, status, datetime.now()])

def init_state():
    defaults = {
        'logged_in': False, 
        'customer_id': None, 
        'result': None, 
        'history': defaultdict(list), 
        'session_txns': defaultdict(lambda: {'count': 0, 'amount': 0}),
        'user_spending_history': defaultdict(float),
        'txn_in_progress': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_velocity(cid):
    now = datetime.now()
    cid = str(cid)
    hist = st.session_state.history.get(cid, [])
    recent_30s = [t for t in hist if (now - t).total_seconds() < 30]
    cnt_30s = len(recent_30s)
    
    time_last = (now - max(hist)).total_seconds() if hist else 3600

    return {
        'txn_count_30s': cnt_30s,
        'time_since_last_txn': time_last
    }

def record_txn(cid):
    cid = str(cid)
    if cid not in st.session_state.history:
        st.session_state.history[cid] = []
    st.session_state.history[cid].append(datetime.now())

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

def get_monthly_spending(cid, cust_data):
    now = datetime.now()
    current_month = now.month
    current_year = now.year

    total_spending = 0.0

    # 1. Calculate from Historical Data (Clean.csv / Feature Engineered)
    if 'CreateDate' in cust_data.columns and len(cust_data) > 0:
        # Avoid SettingWithCopyWarning
        temp_df = cust_data.copy()
        temp_df['CreateDate'] = pd.to_datetime(temp_df['CreateDate'], errors='coerce')

        # Filter for current month/year
        monthly_mask = (temp_df['CreateDate'].dt.month == current_month) & \
                       (temp_df['CreateDate'].dt.year == current_year)

        # Select appropriate amount column
        if 'transaction_amount' in temp_df.columns:
            amt_col = 'transaction_amount'
        elif 'AmountInAed' in temp_df.columns:
            amt_col = 'AmountInAed'
        else:
            amt_col = 'Amount'

        total_spending += temp_df.loc[monthly_mask, amt_col].sum()

    # 2. Calculate from Recent Session History (transaction_history.csv)
    file_name = 'transaction_history.csv'
    if os.path.isfile(file_name):
        try:
            recent_df = pd.read_csv(file_name)
            # Filter by CID
            recent_df = recent_df[recent_df['CustomerID'].astype(str) == str(cid)]

            if 'Timestamp' in recent_df.columns and 'Amount' in recent_df.columns:
                recent_df['Timestamp'] = pd.to_datetime(recent_df['Timestamp'], errors='coerce')
                monthly_mask = (recent_df['Timestamp'].dt.month == current_month) & \
                               (recent_df['Timestamp'].dt.year == current_year)

                # Filter by Approved status
                if 'Status' in recent_df.columns:
                     status_mask = recent_df['Status'].astype(str).str.contains('Approved', case=False)
                     monthly_mask = monthly_mask & status_mask

                total_spending += recent_df.loc[monthly_mask, 'Amount'].sum()
        except Exception as e:
            print(f"Error reading transaction_history.csv: {e}")

    return total_spending

# ---------------------- LOGIN PAGE ----------------------

def login_page():
    st.title("🏦 Banking Fraud Detection System")
    st.subheader("Login")
    df = load_data()
    if df is None:
        if not run_pipeline(): return
        st.rerun()
    customers = sorted([str(c) for c in df['CustomerId'].dropna().unique()])
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Select Customer ID")
        cid = st.selectbox("Customer ID", customers)
        pwd = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if pwd == "12345":
                st.session_state.logged_in = True
                st.session_state.customer_id = cid
                st.rerun()
            else:
                st.error("Invalid password")
        st.info("Password: 12345")

# ---------------------- DASHBOARD ----------------------

def dashboard():
    from backend.hybrid_decision import make_decision
    from backend.rule_engine import calculate_all_limits
    from backend.model_training import load_model

    cid = str(st.session_state.customer_id)
    df = load_data()
    amt_col = 'transaction_amount' if 'transaction_amount' in df.columns else 'Amount'
    cust_data = df[df['CustomerId'].astype(str) == cid]

    # Initialize session
    if cid not in st.session_state.session_txns:
        st.session_state.session_txns[cid] = {'count': 0, 'amount': 0}

    current_session_count = st.session_state.session_txns[cid]['count']
    csv_count = len(cust_data)
    total_txns = csv_count + current_session_count

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Customer ID:** {cid}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.customer_id = None
        st.session_state.result = None
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Customer Statistics")
    if len(cust_data) > 0:
        avg = cust_data[amt_col].mean()
        max_amt = cust_data[amt_col].max()
        std = cust_data[amt_col].std() if len(cust_data) > 1 else 0
        st.sidebar.markdown(f"**Average Transaction:** AED {avg:,.2f}")
        st.sidebar.markdown(f"**Max Transaction:** AED {max_amt:,.2f}")
        st.sidebar.metric("Total Transactions", total_txns, delta=f"+{current_session_count} New" if current_session_count > 0 else None)

        limits = calculate_all_limits(avg, std)
        st.sidebar.markdown(f"**O - Own Account:** AED {limits['O']:,.2f}")
        st.sidebar.markdown(f"**I - Ajman:** AED {limits['I']:,.2f}")
        st.sidebar.markdown(f"**L - UAE:** AED {limits['L']:,.2f}")
        st.sidebar.markdown(f"**Q - Quick:** AED {limits['Q']:,.2f}")
        st.sidebar.markdown(f"**S - Overseas:** AED {limits['S']:,.2f}")

    # ---------------- TRANSACTION FORM ----------------
    st.title("Fraud Detection Dashboard")
    st.subheader("Step 1: Select Account")
    accounts = [str(a) for a in cust_data['FromAccountNo'].dropna().unique()] if 'FromAccountNo' in cust_data.columns else ["Default"]
    account = st.selectbox("Select Account", accounts, label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("Step 2: Transaction Details")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Transaction Amount (AED)**")
        amount = st.number_input("", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0)
    with c2:
        st.markdown("**Transfer Type**")
        types = {'O': 'O - Own Account', 'I': 'I - Within Ajman', 'L': 'L - Within UAE', 'Q': 'Q - Quick Remittance', 'S': 'S - Overseas'}
        t_type = st.selectbox("", list(types.keys()), format_func=lambda x: types[x])
    with c3:
        st.markdown("**Bank Country**")
        countries = ['UAE','USA','UK','India','Pakistan','Philippines','Egypt','Other']
        country = st.selectbox("", countries)

    st.markdown("---")
    st.subheader("Step 3: Process Transaction")
    if st.button("Process Transaction"):
        if not st.session_state.txn_in_progress:
            st.session_state.txn_in_progress = True
            record_txn(cid)
            
            # --- BURST DETECTION ---
            vel = get_velocity(cid)
            BURST_COUNT_THRESHOLD = 5
            is_burst = vel['txn_count_30s'] > BURST_COUNT_THRESHOLD

            # --- USER STATS & ML FEATURE PREP ---
            # Calculate true monthly spending
            current_spending = get_monthly_spending(cid, cust_data)

            current_type_txns = cust_data[cust_data['TransferType'] == t_type]
            specific_avg = current_type_txns[amt_col].mean() if len(current_type_txns) > 0 else (cust_data[amt_col].mean() if len(cust_data) > 0 else 5000)
            specific_std = current_type_txns[amt_col].std() if len(current_type_txns) > 1 else (cust_data[amt_col].std() if len(cust_data) > 1 else 2000)
            specific_max = current_type_txns[amt_col].max() if len(current_type_txns) > 0 else (cust_data[amt_col].max() if len(cust_data) > 0 else 15000)

            total_txns_count = len(cust_data)
            intl_ratio = ((cust_data['TransferType'] == 'S').sum() / total_txns_count) if total_txns_count>0 else 0
            high_risk_ratio = (((cust_data['TransferType'] == 'S') | (cust_data['TransferType'] == 'Q')).sum() / total_txns_count) if total_txns_count>0 else 0

            user_stats = {
                'user_avg_amount': specific_avg,
                'user_std_amount': specific_std,
                'user_max_amount': specific_max,
                'user_txn_frequency': total_txns_count,
                'user_international_ratio': intl_ratio,
                'user_high_risk_txn_ratio': high_risk_ratio,
                'current_month_spending': current_spending
            }

            txn = {'amount': amount, 'transfer_type': t_type, 'bank_country': country, **vel}

            # --- ML / Hybrid Decision ---
            from backend.model_training import load_model
            try:
                model, feats = load_model()
                ml_result = make_decision(txn, user_stats, model, feats)
            except:
                ml_result = make_decision(txn, user_stats)
            ml_result['amount'] = amount

            # --- MERGE BURST FLAG ---
            if is_burst:
                ml_result['is_fraud'] = True
                ml_result['reasons'] = ml_result.get('reasons', []) + ["Burst Transaction: High frequency in short time"]
                ml_result['risk_score'] = max(ml_result.get('risk_score', 0), 0.9)

            st.session_state.result = ml_result
            st.session_state.txn_in_progress = False
            st.rerun()

    # ---------------- STEP 4: TRANSACTION RESULT ----------------
    st.markdown("---")
    st.subheader("Step 4: Transaction Result")
    if st.session_state.result:
        r = st.session_state.result
        if r['is_fraud']:
            st.error("⚠️ Transaction Flagged!")
            for reason in r['reasons']:
                st.warning(f"- {reason}")
            c1,c2 = st.columns(2)
            with c1:
                if st.button("Approve Transaction (Force)"):
                    save_transaction_to_csv(cid, r['amount'], t_type, "Force Approved")
                    st.session_state.user_spending_history[cid] += r['amount']
                    st.session_state.session_txns[cid]['count'] += 1
                    st.session_state.result = None
                    st.rerun()
            with c2:
                if st.button("Reject Transaction"):
                    st.error("Transaction Rejected!")
                    st.session_state.result = None
                    st.rerun()
        else:
            st.success("SAFE TRANSACTION ✅")
            if st.button("Process Another Transaction"):
                save_transaction_to_csv(cid, r['amount'], t_type, "Approved")
                st.session_state.user_spending_history[cid] += r['amount']
                st.session_state.session_txns[cid]['count'] += 1
                st.session_state.result = None
                st.rerun()

# ---------------------- MAIN ----------------------

def main():
    init_state()
    if not os.path.exists('data/Clean.csv'):
        st.warning("Please ensure data/Clean.csv exists")
        return
    if not run_pipeline(): return
    if st.session_state.logged_in:
        dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()
