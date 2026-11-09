# AML Sentinel AI — End-to-End AI Financial Intelligence

Repository: https://github.com/VirajBarapatre/AML-Sentinel-AI

## Overview

AML Sentinel is an Anti-Money Laundering (AML) platform designed to detect,
visualize, and manage global financial risks. The system integrates a full-stack
data pipeline — from synthetic transaction generation to unsupervised anomaly
detection and geospatial intelligence.

This is a high-fidelity simulation developed for financial technology research and
portfolio demonstration. All data generated is synthetic.

## Project Structure

The project is organized into a modular pipeline:

- `database_setup.py` — initializes a SQLite RDBMS and generates 2,000+ global users
  with high-fidelity geospatial coordinates
- `feature_eng.py` — processes raw transaction logs into behavioral vectors (e.g.
  velocity, structuring indicators, jurisdictional risk)
- `model_engine.py` — the AI core; uses an Isolation Forest model to identify
  statistical outliers in financial behavior
- `watchlists.py` — manages global sanctions lists and PEP (Politically Exposed
  Persons) screening logic
- `dashboard.py` — interactive Streamlit frontend with high-resolution PyDeck
  heatmaps and case management tools

## Key Features

### 1. Advanced Geospatial Heatmapping
Dual-layer PyDeck implementation to visualize risk density:
- Logarithmic intensity scaling, so emerging risk in smaller markets (e.g. Mumbai)
  isn't drowned out by high-volume hubs (e.g. Luxembourg)
- Interactive tooltips showing User IDs, Risk Scores, and Last Activity Timestamps

### 2. AI-Driven Detection Engine
Goes beyond simple rule-based thresholds to flag:
- Structuring / smurfing — multiple transactions kept just under reporting limits
- Rapid outflows — high-velocity transfers to offshore jurisdictions

### 3. Case Management Workflow
- Search by User ID to pull a complete transaction "Case File"
- Update alert statuses (Pending → Under Review → SAR) directly from the UI, with
  database persistence

## Tech Stack

- Language: Python 3.9+
- Machine Learning: Scikit-Learn (Isolation Forest)
- Data Science: Pandas, NumPy
- Database: SQLite3
- Visuals: PyDeck (Deck.gl), Streamlit
- Map Tiles: CartoDB (Dark Matter)

## Quick Start

```
pip install streamlit pandas scikit-learn pydeck
python src/database_setup.py
python src/model_engine.py
streamlit run src/dashboard.py
```

## Author

Developed by Viraj Barapatre. GitHub: https://github.com/VirajBarapatre
