"""
producer.py
Rejoue le dataset creditcard.csv comme un flux de transactions en direct,
en envoyant chaque ligne sous forme de message JSON sur un topic Kafka.

Usage:
    python producer.py

Dépendances:
    pip install kafka-python pandas
"""

import json
import time
import uuid
import pandas as pd
from kafka import KafkaProducer

# ---- Configuration ----
CSV_PATH ="/mnt/d/machine learning/bank/creditcard.csv"  # chemin du dataset brute
KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
KAFKA_TOPIC = "transactions"
DELAY_SECONDS = 0.2  # délai entre deux messages pour simuler un flux réel


def main():
    df = pd.read_csv(CSV_PATH)

    # Colonnes nécessaires pour reconstruire les features côté consumer :
    # Time, V1..V28, Amount. On garde Class uniquement comme vérité terrain
    # (à ne PAS utiliser pour la prédiction, juste pour évaluer le système après coup).
    v_cols = [f"V{i}" for i in range(1, 29)]
    required_cols = ["Time", "Amount"] + v_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Envoi de {len(df)} transactions vers le topic '{KAFKA_TOPIC}'...")

    for _, row in df.iterrows():
        message = {
            "transaction_id": str(uuid.uuid4()),
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
            **{c: float(row[c]) for c in v_cols},
            "Class": int(row["Class"]) if "Class" in row and pd.notna(row["Class"]) else None,
        }
        producer.send(KAFKA_TOPIC, value=message)
        time.sleep(DELAY_SECONDS)

    producer.flush()
    producer.close()
    print("Terminé.")


if __name__ == "__main__":
    main()