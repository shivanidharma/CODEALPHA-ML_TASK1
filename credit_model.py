# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("german.data-numeric.csv",sep=';',header=None)

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df.iloc[:, -1].value_counts())

# =========================
# 3. SPLIT FEATURES & TARGET
# =========================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    
)

# =========================
# 5. FEATURE SCALING (FOR LOGISTIC REGRESSION)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. MODEL 1: LOGISTIC REGRESSION
# =========================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

# =========================
# 7. MODEL 2: RANDOM FOREST
# =========================
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# =========================
# 8. EVALUATION FUNCTION
# =========================
def evaluate(name, y_test, y_pred, y_prob):
    print("\n========================")
    print(name)
    print("========================")

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =========================
# 9. RESULTS
# =========================
evaluate("Logistic Regression", y_test, log_pred, log_prob)
evaluate("Random Forest", y_test, rf_pred, rf_prob)

# =========================
# 10. FEATURE IMPORTANCE (RF)
# =========================
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()