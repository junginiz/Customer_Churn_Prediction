# Customer_Churn_Prediction
The project predicts whether a customer is likely to churn (leave) using historical data from a telecom company. It applies machine learning models and business analytics to help customer retention strategies, which are crucial use case for the e-commerce and subscription-based businesses.

Key Features
Exploratory Data Analysis (EDA): Uncovered churn patterns using contract type, monthly charges, and tenure.
Modeling: Built and compared models using Logistic Regression, Random Forest, and XGBoost.
Feature Engineering: Handled missing data, encoded categorical features, and standardized numerical values.
Performance Metrics: Evaluated models using accuracy, precision, recall, and confusion matrix.
Deployment: A simple Streamlit web app for real-time churn prediction.

 Results
The best-performing model was XGBoost, achieving:
Accuracy: ~91%
Precision & Recall: Balanced for both churn and non-churn classes
Top Features: tenure, MonthlyCharges, Contract, and InternetService
