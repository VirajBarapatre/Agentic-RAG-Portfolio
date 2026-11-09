# AI Dynamic Pricing Simulator

Repository: https://github.com/VirajBarapatre/dynamic-pricing-simulator
Live Demo: https://dynamic-pricing-simulator.onrender.com/ (Render free tier — may take
~30s cold start)
License: MIT

## Overview

An AI-powered simulator that predicts demand, revenue, and profit for products across
multiple categories and recommends the optimal price using machine learning.

## Features

- Predicts demand, revenue, and profit for a user-input price
- Recommends the optimal product price using an ML model
- Interactive web interface built with Flask + Tailwind CSS
- Visualizes demand and profit curves with Chart.js
- Dataset generator that simulates real-world market data (20+ products per category)
- Trained with a Random Forest model for accuracy and fast predictions

## Project Structure

```
ai-dynamic-pricing-simulator/
├── app.py                  # Main Flask app
├── extended_retail_data.py # Generates synthetic dataset
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── data/
│   └── extended_retail_data.csv
└── templates/
    └── index.html           # Frontend page
```

## Tech Stack

- Backend: Flask (Python)
- Machine Learning: scikit-learn, pandas, numpy
- Frontend: Tailwind CSS, Chart.js
- Deployment: Render

## How to Run Locally

```
git clone https://github.com/VirajBarapatre/dynamic-pricing-simulator.git
cd dynamic-pricing-simulator
pip install -r requirements.txt
python dataset_generator.py
python app.py
```

App runs at http://127.0.0.1:5000

## Future Improvements

- User authentication for business accounts
- Upload & train on real-world pricing datasets
- Explore Reinforcement Learning for dynamic pricing updates
- Deploy on AWS/GCP/Azure for scalability

## Author

Developed by Viraj Barapatre. GitHub: https://github.com/VirajBarapatre
