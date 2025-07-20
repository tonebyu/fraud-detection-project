# Fraud Detection System for E-Commerce & Banking 🛡️💳

This project builds robust and explainable machine learning models to detect fraudulent transactions in both e-commerce and credit card datasets. It includes advanced feature engineering, geolocation enrichment, and interpretable model diagnostics using SHAP.

🚀 **Goals:**
- Clean and analyze transactional data from multiple sources
- Engineer time-based, frequency-based, and geolocation-aware features
- Handle extreme class imbalance using modern resampling techniques
- Build and compare Logistic Regression and Ensemble models (XGBoost / Random Forest)
- Explain model behavior using SHAP for trustworthy decision-making

📦 **Deliverables:**
- Notebooks with clear Task-based breakdown
- Clean modular codebase for reproducibility
- Rich visualizations and interpretability reports
- Deployment-ready GitHub repository

📁 **Datasets Used:**
- `Fraud_Data.csv` – E-commerce transactions
- `creditcard.csv` – Bank transaction data
- `IpAddress_to_Country.csv` – IP address geo-resolution
:: Create virtual environment
python -m venv .venv

:: Activate it
.venv\Scripts\activate

:: Install libraries
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap ipykernel

:: Freeze requirements
pip freeze > requirements.txt
