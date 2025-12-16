# Banking Fraud Detection System

## Overview
A Streamlit-based fraud detection system that uses Machine Learning (Isolation Forest) and rule-based validation to detect anomalous banking transactions.

## Project Structure
```
├── app.py                    # Main Streamlit application
├── backend/
│   ├── utils.py              # Helper functions and constants
│   ├── rule_engine.py        # Dynamic threshold calculations
│   ├── feature_engineering.py # Feature creation for ML
│   ├── model_training.py     # Isolation Forest model
│   └── hybrid_decision.py    # Combined ML + Rules decision
├── data/
│   ├── Clean.csv             # Clean transaction data
│   └── feature_Engineered.csv # Processed features
├── models/
│   └── isolation_forest.pkl  # Trained model
└── PROJECT_LOGIC.md          # Detailed logic documentation
```

## Running the App
```bash
streamlit run app.py --server.port 5000
```

## Features
- Login with Customer ID and password (12345)
- Transaction processing with fraud detection
- Dynamic limits per transfer type (S, I, L, Q, O)
- Approve/Reject for anomaly transactions
- Customer statistics in sidebar

## Transfer Types
- O: Own Account (lowest risk)
- I: Within Ajman (low risk)
- L: Within UAE (low risk)
- Q: Quick Remittance (medium risk)
- S: Overseas (high risk)
