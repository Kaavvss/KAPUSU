# KAPUSU
KAPUSU is a network-aware financial stress detection and early intervention mini prototype.

## What it does
- Simulates microfinance borrower networks
- Detects individual distress, network exposure, and local economic stress
- Uses graph-engineered features plus a machine learning classifier
- Provides an interactive Streamlit dashboard

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data generation
The app automatically generates the required synthetic data in `data/` on first run if the CSV files do not exist.

## Demo scenario
Use the sidebar controls in the app to simulate a financial shock for a borrower and watch risk spread through the borrower network.
