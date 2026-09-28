"""
consumer.py
Lit le flux de transactions depuis Kafka, applique le modèle de détection
de fraude entraîné dans le notebook (fraud_model.pkl + scaler.pkl), et
sauvegarde dans MongoDB les transactions prédites comme frauduleuses.

Dépendances:
    pip install pyspark pymongo joblib pandas numpy scikit-learn
"""

import os
import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pymongo import MongoClient

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 pyspark-shell"
)

# ---- Chargement du modèle et du scaler entraînés dans le notebook ----
MODEL_PATH = "fraud_model.pkl"   # chemin vers le .pkl sauvegardé par joblib
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Ordre exact des features attendu par le modèle (celui du notebook) :
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = V_COLS + ["hour_sin", "hour_cos", "Amount_standarise"]

# ---- Connexion MongoDB ----
client = MongoClient("mongodb://localhost:27017")
db = client["fraud_detection"]
collection_fraudes = db["fraudes_detectees"]      # fraudes uniquement (alertes)
collection_toutes = db["toutes_transactions"]      # toutes les transactions (analyse)

collection_toutes.create_index("transaction_id")
collection_toutes.create_index("Time")

# ---- Connexion Spark ----
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Schéma des messages envoyés par producer.py
fields = [
    StructField("transaction_id", StringType()),
    StructField("Time", DoubleType()),
    StructField("Amount", DoubleType()),
] + [StructField(v, DoubleType()) for v in V_COLS] + [
    StructField("Class", IntegerType())  # vérité terrain, non utilisée pour la prédiction
]
schema = StructType(fields)

# ---- Lire depuis Kafka ----
raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "127.0.0.1:9092")
    .option("subscribe", "transactions")
    .load()
)

transactions = raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")


def predire_et_sauvegarder(batch_df, batch_id):
    """Applique le modèle sur le micro-batch et sauvegarde les fraudes prédites."""
    pdf = batch_df.toPandas()
    if pdf.empty:
        return

    # 1. Recréer les features exactement comme à l'entraînement
    pdf["Amount_log"] = np.log1p(pdf["Amount"])

    start_date = pd.Timestamp("2026-01-01")
    pdf["timestamp"] = start_date + pd.to_timedelta(pdf["Time"], unit="s")
    pdf["hour"] = pdf["timestamp"].dt.hour
    pdf["hour_sin"] = np.sin(2 * np.pi * pdf["hour"] / 24)
    pdf["hour_cos"] = np.cos(2 * np.pi * pdf["hour"] / 24)

    # 2. Standardiser Amount_log avec le scaler entraîné (univarié)
    pdf["Amount_standarise"] = scaler.transform(pdf[["Amount_log"]])

    # 3. Construire la matrice de features dans le même ordre que l'entraînement
    X_batch = pdf[FEATURE_ORDER]

    # 4. Prédire
    pdf["prediction"] = model.predict(X_batch)
    pdf["proba_fraude"] = model.predict_proba(X_batch)[:, 1]

    # 5. Sauvegarder TOUTES les transactions du batch (pour l'analyse : taux de
    #    fraude, comparaisons fraude vs non-fraude, distributions, etc.)
    colonnes_a_garder = [
        "transaction_id", "Time", "Amount", "hour_sin", "hour_cos",
        "prediction", "proba_fraude", "Class",
    ]
    docs_toutes = pdf[colonnes_a_garder].to_dict("records")
    collection_toutes.insert_many(docs_toutes)

    # 6. Sauvegarder séparément les fraudes prédites (pour les alertes rapides)
    fraudes = pdf[pdf["prediction"] == 1]
    if not fraudes.empty:
        docs_fraudes = fraudes[
            ["transaction_id", "Time", "Amount", "proba_fraude", "Class"]
        ].to_dict("records")
        collection_fraudes.insert_many(docs_fraudes)
        print(f"[batch {batch_id}] {len(docs_fraudes)} fraude(s) détectée(s) et sauvegardée(s).")

    print(f"[batch {batch_id}] {len(docs_toutes)} transaction(s) au total sauvegardée(s).")


query = transactions.writeStream.foreachBatch(predire_et_sauvegarder).start()
query.awaitTermination()