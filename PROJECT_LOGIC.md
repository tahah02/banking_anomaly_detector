# Banking Fraud Detection System - Complete Logic Documentation

## Overview
Yeh system banking transactions ko analyze karta hai aur fraud/anomalies detect karta hai using:
1. **Machine Learning (Isolation Forest)** - Unusual patterns detect karta hai
2. **Rule Engine** - Dynamic thresholds check karta hai per transfer type

---

## File Structure

```
├── app.py                    # Main Streamlit application
├── backend/
│   ├── utils.py              # Helper functions aur constants
│   ├── rule_engine.py        # Dynamic threshold calculations
│   ├── feature_engineering.py # Data ko ML-ready features mein convert
│   ├── model_training.py     # Isolation Forest model training
│   └── hybrid_decision.py    # ML + Rules combine karke decision
├── data/
│   ├── Clean.csv             # Original clean transaction data
│   └── feature_Engineered.csv # Processed features wala data
├── models/
│   └── isolation_forest.pkl  # Trained ML model
└── PROJECT_LOGIC.md          # Yeh documentation file
```

---

## 1. utils.py - Helper Functions

### Purpose:
Basic utility functions aur constants define karta hai.

### Key Components:

```python
# File paths
get_clean_csv_path()           # data/Clean.csv path
get_feature_engineered_path()  # data/feature_Engineered.csv path  
get_model_path()               # models/isolation_forest.pkl path

# Transfer Type Mappings
TRANSFER_TYPE_MAPPING = {
    'S': 'Overseas',    # International transfer - HIGH RISK
    'I': 'Ajman',       # Local Ajman transfer - LOW RISK
    'L': 'UAE',         # Within UAE transfer - LOW RISK
    'Q': 'Quick',       # Quick remittance - MEDIUM RISK
    'O': 'Own'          # Own account transfer - LOWEST RISK
}

# Encoded values for ML model
TRANSFER_TYPE_ENCODED = {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}

# Risk scores (0 to 1, higher = more risky)
TRANSFER_TYPE_RISK = {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}
```

---

## 2. rule_engine.py - Dynamic Thresholds

### Purpose:
Har transfer type ke liye dynamic spending limits calculate karta hai.

### Formula:
```
Threshold = user_avg_amount + (multiplier × user_std_amount)
Final_Limit = max(Threshold, minimum_floor)
```

### Transfer Type Multipliers:
```python
TRANSFER_MULTIPLIERS = {
    'S': 2.0,   # Overseas - Strictest (2x std dev)
    'Q': 2.5,   # Quick - Strict (2.5x std dev)
    'L': 3.0,   # UAE - Medium (3x std dev)
    'I': 3.5,   # Ajman - Relaxed (3.5x std dev)
    'O': 4.0    # Own Account - Most Relaxed (4x std dev)
}
```

### Minimum Floors:
```python
TRANSFER_MIN_FLOORS = {
    'S': 5000,  # Overseas minimum AED 5000
    'Q': 3000,  # Quick minimum AED 3000
    'L': 2000,  # UAE minimum AED 2000
    'I': 1500,  # Ajman minimum AED 1500
    'O': 1000   # Own Account minimum AED 1000
}
```

### Example:
Agar user ka:
- Average = AED 1000
- Std Dev = AED 500

**Overseas (S) Limit:**
- Statistical = 1000 + (2.0 × 500) = 2000
- Final = max(2000, 5000) = **AED 5000**

**Own Account (O) Limit:**
- Statistical = 1000 + (4.0 × 500) = 3000
- Final = max(3000, 1000) = **AED 3000**

### Logic:
- **Overseas (S)** strictest kyunki high risk hai
- **Own Account (O)** most relaxed kyunki apne hi account mein transfer hai

---

## 3. feature_engineering.py - Feature Creation

### Purpose:
Raw transaction data ko ML model ke liye useful features mein convert karta hai.

### Features Created:

#### Transaction Features:
| Feature | Description |
|---------|-------------|
| `transaction_amount` | Transaction ki amount |
| `flag_amount` | 1 agar Overseas (S), warna 0 |
| `transfer_type_encoded` | S=4, Q=3, L=2, I=1, O=0 |
| `transfer_type_risk` | Risk score (0-1) |
| `channel_encoded` | Channel ID encoded |

#### Time Features:
| Feature | Description |
|---------|-------------|
| `hour` | Transaction ka hour (0-23) |
| `day_of_week` | 0=Monday, 6=Sunday |
| `is_weekend` | 1 agar Saturday/Sunday |
| `is_night` | 1 agar 10pm-6am |

#### User Behavior Features:
| Feature | Description |
|---------|-------------|
| `user_avg_amount` | User ki average transaction |
| `user_std_amount` | User ki standard deviation |
| `user_max_amount` | User ki maximum transaction |
| `user_txn_frequency` | User ki total transactions |
| `deviation_from_avg` | Current amount - user average |
| `amount_to_max_ratio` | Current / max amount |
| `intl_ratio` | User ki international transactions ka ratio |

#### Velocity Features (Burst Detection):
| Feature | Description |
|---------|-------------|
| `time_since_last` | Pichli transaction se time (seconds) |
| `recent_burst` | 1 agar last txn < 5 min ago |
| `txn_count_10min` | Last 10 min mein transactions |
| `txn_count_1hour` | Last 1 hour mein transactions |

---

## 4. model_training.py - Isolation Forest

### Purpose:
Anomaly detection ke liye Isolation Forest model train karta hai.

### How Isolation Forest Works:

```
Normal transactions = bahut transactions similar hoti hain
Anomalies = alag patterns, jaldi isolate ho jaati hain

Algorithm:
1. Random features select karo
2. Random split points se data divide karo
3. Jo transactions jaldi isolate ho = ANOMALY
4. Jo transactions der se isolate ho = NORMAL
```

### Model Configuration:
```python
IsolationForest(
    n_estimators=100,     # 100 decision trees
    contamination=0.05,   # 5% transactions anomaly expect
    random_state=42,      # Reproducible results
    n_jobs=-1             # All CPU cores use karo
)
```

### Output:
- **Prediction = 1**: Normal transaction
- **Prediction = -1**: Anomaly detected

### Decision Function:
- Negative score = anomaly (lower = more anomalous)
- Positive score = normal

---

## 5. hybrid_decision.py - Combined Decision

### Purpose:
ML model aur Rule Engine dono ko combine karke final decision leta hai.

### Decision Flow:

```
                    Transaction Input
                          ↓
              ┌───────────────────────┐
              │   Prepare Features    │
              └───────────┬───────────┘
                          ↓
         ┌────────────────┴────────────────┐
         ↓                                 ↓
┌─────────────────┐              ┌─────────────────┐
│   ML Model      │              │   Rule Engine   │
│ (Isolation      │              │ (Dynamic        │
│  Forest)        │              │  Thresholds)    │
└────────┬────────┘              └────────┬────────┘
         ↓                                 ↓
    Velocity                          Amount
    Anomaly?                          > Limit?
         ↓                                 ↓
         └────────────────┬────────────────┘
                          ↓
                   is_fraud = TRUE if
                   (ML Anomaly OR Rule Violation)
```

### Anomaly Types:
1. **ML Velocity Anomaly**: Jab ML detect kare ke transaction pattern unusual hai (burst transactions)
2. **Rule Violation**: Jab amount threshold se zyada ho

---

## 6. app.py - Streamlit Application

### Purpose:
User interface provide karta hai fraud detection system ke liye.

### Session State Variables:
```python
logged_in          # User logged in hai ya nahi
customer_id        # Current customer ID
result             # Last transaction ka result
history            # Transaction timestamps per customer
session_txns       # Session mein count aur amount track
```

### Features:

#### Login Page:
- Customer ID dropdown (data se auto-populate)
- Generic password: **12345**
- All customers same password

#### Dashboard:
1. **Sidebar:**
   - Customer ID display
   - Session Stats (count + total amount)
   - Transfer Type Limits (S, I, L, Q, O)
   - Logout button

2. **Transaction Form:**
   - Account selection
   - Amount input
   - Transfer Type (S/I/L/Q/O)
   - Bank Country

3. **Result Section:**
   - Safe: Green success message
   - Anomaly: Red alert with reasons
   - Approve/Reject buttons for ALL anomalies

### Velocity Tracking:
```python
def get_velocity(customer_id):
    # Last 1 hour transactions filter
    # Count in 10 min window
    # Count in 1 hour window
    # Time since last transaction
```

### Session Tracking:
```python
def record_txn(customer_id, amount):
    # Timestamp record karo
    # Session count ++ 
    # Session amount += amount
```

---

## Data Flow Summary

```
1. User Login
      ↓
2. Select Account + Enter Transaction Details
      ↓
3. Click "Analyze Transaction"
      ↓
4. Get User Stats from Historical Data
   - Average amount
   - Standard deviation
   - Max amount
   - Transaction frequency
      ↓
5. Get Velocity Info from Session
   - Transactions in 10 min
   - Transactions in 1 hour
   - Time since last
      ↓
6. Prepare Features for ML Model
      ↓
7. ML Model Prediction
   - Normal (1) or Anomaly (-1)
      ↓
8. Rule Engine Check
   - Amount vs Dynamic Threshold
      ↓
9. Combined Decision
   - is_fraud = ML_Anomaly OR Rule_Violation
      ↓
10. Display Result
    - Safe → Confirm button
    - Anomaly → Approve/Reject buttons
      ↓
11. Record Transaction (if approved/confirmed)
    - Update session count
    - Update session total amount
```

---

## Why These Choices?

### Why Isolation Forest?
- Unsupervised learning (no labels needed)
- Good for anomaly detection
- Handles high-dimensional data
- Fast training and prediction

### Why Dynamic Thresholds?
- Each user ka different spending pattern
- Transfer type ke mutabiq risk adjust
- Minimum floors prevent false negatives for new users

### Why Hybrid Approach?
- ML catches complex patterns (burst transactions)
- Rules catch simple violations (amount limits)
- Combination gives better accuracy

### Why Session Tracking?
- Real-time burst detection
- User ko apni session activity dikhana
- Transaction velocity monitoring

---

## Transfer Type Risk Analysis

| Type | Risk | Multiplier | Min Floor | Reason |
|------|------|------------|-----------|--------|
| S | HIGH | 2.0x | 5000 | International = more fraud risk |
| Q | MEDIUM | 2.5x | 3000 | Quick remittance = moderate risk |
| L | LOW | 3.0x | 2000 | Within UAE = regulated |
| I | LOW | 3.5x | 1500 | Within Ajman = local, safe |
| O | LOWEST | 4.0x | 1000 | Own account = safest |

---

## Conclusion

Yeh system:
1. **Simple** - Easy to understand code
2. **Effective** - ML + Rules combination
3. **User Friendly** - Clear UI with all info in sidebar
4. **Real-time** - Session tracking aur velocity monitoring
5. **Customizable** - Per user dynamic limits
