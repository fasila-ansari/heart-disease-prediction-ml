# ============================================================
# Heart Disease Prediction using Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# Create images folder automatically
# ============================================================

os.makedirs("images", exist_ok=True)

print("====================================")
print("Heart Disease Prediction Project")
print("====================================")

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("heart.csv")

print("\nDataset Loaded Successfully")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# Check Missing Values
# ============================================================

print("\nChecking missing values...")

print(df.isnull().sum())

# ============================================================
# Data Description
# ============================================================

print("\nDataset statistics:")

print(df.describe())

# ============================================================
# Correlation Heatmap
# ============================================================

print("\nGenerating correlation heatmap...")

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.savefig("images/correlation_heatmap.png")

plt.show()

# ============================================================
# Feature / Target Split
# ============================================================

X = df.drop("target", axis=1)

y = df["target"]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# Model Training and Comparison
# ============================================================

print("\nTraining Multiple Machine Learning Models...")

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"\n{name} Accuracy:", accuracy)

# ============================================================
# Model Accuracy Comparison Plot
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.savefig("images/model_comparison.png")

plt.show()

# ============================================================
# Train Best Model (Random Forest)
# ============================================================

print("\nTraining Final Model (Random Forest)...")

final_model = RandomForestClassifier()

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# ============================================================
# Feature Importance
# ============================================================

importance = final_model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Feature Importance in Heart Disease Prediction")

plt.savefig("images/feature_importance.png")

plt.show()

print("\nProject Completed Successfully!")