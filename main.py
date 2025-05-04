# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load the Dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 3. Basic Cleaning
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 4. Label Encoding
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col])

# 5. EDA - Optional for Jupyter Notebook
# sns.countplot(x='Churn', data=df)
# plt.title("Churn Distribution")
# plt.show()

# 6. Feature Matrix and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 7. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. Model Training and Evaluation
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔍 Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Logistic Regression
evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")

# Random Forest
evaluate_model(RandomForestClassifier(n_estimators=100), "Random Forest")

# XGBoost
evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost")

# 10. Feature Importance (for XGBoost)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
importances = xgb.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()
