"""
modele.py
Script d'entraînement canonique — reproduit exactement le pipeline du
notebook (feature engineering, split, SMOTE, LogisticRegression) pour
produire fraud_model.pkl et scaler.pkl utilisés en production par
consumer.py.

IMPORTANT: l'ordre des features ici doit rester identique à FEATURE_ORDER
dans consumer.py : V1..V28, hour_sin, hour_cos, Amount_standarise.

Dépendances:
    pip install pandas numpy scikit-learn imbalanced-learn joblib
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ---- Configuration ----
CSV_PATH = "/mnt/d/machine learning/bank/creditcard.csv" # dataset brut (Time, V1..V28, Amount, Class)
MODEL_OUT = "fraud_model.pkl"
SCALER_OUT = "scaler.pkl"

V_COLS = [f"V{i}" for i in range(1, 29)]


def main():
    df = pd.read_csv(CSV_PATH)

    # 1. Feature engineering — identique au notebook
    df["Amount_log"] = np.log1p(df["Amount"])

    start_date = pd.Timestamp("2026-01-01")
    df["timestamp"] = start_date + pd.to_timedelta(df["Time"], unit="s")
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 2. X / y
    X = df.drop(columns=["Class", "Time", "Amount", "hour"])
    y = df["Class"]

    # 3. Split stratifié (comme le notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Standardisation d'Amount_log -> Amount_standarise
    scaler = StandardScaler()
    X_train["Amount_standarise"] = scaler.fit_transform(X_train[["Amount_log"]])
    X_test["Amount_standarise"] = scaler.transform(X_test[["Amount_log"]])

    # 5. Colonnes finales, dans l'ordre attendu par consumer.py
    feature_order = V_COLS + ["hour_sin", "hour_cos", "Amount_standarise"]
    X_train_final = X_train[feature_order]
    X_test_final = X_test[feature_order]

    # 6. SMOTE uniquement sur le train (pas de class_weight en plus)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)

    # 7. Entraînement
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    # 8. Évaluation
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]

    print("=== Résultats ===")
    print(classification_report(y_test, y_pred))
    print("=== Matrice de confusion ===")
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # 9. Sauvegarde (joblib, cohérent avec consumer.py)
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"Modèle sauvegardé dans {MODEL_OUT}, scaler dans {SCALER_OUT}")


if __name__ == "__main__":
    main()