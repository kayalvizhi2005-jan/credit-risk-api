import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

print("Loading data...")
os.chdir(r'C:\Users\91807\Downloads\credit-risk-project')
df = pd.read_csv(r'cs-training.csv\cs-training.csv', index_col=0)

# Clean data
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
df = df[df['age'] > 18]
df = df[df['age'] < 100]
df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1]
df = df.dropna()

# Prepare features
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train model
print("Training model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_bal, y_train_bal)

# Save model
os.chdir(r'C:\Users\91807\Downloads\credit-risk-api')
joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')

print("✓ Model saved as credit_risk_model.pkl")
print(f"✓ Features: {list(X.columns)}")