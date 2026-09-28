# 🛡️ FraudGuard — Détection de fraude bancaire en temps réel

Pipeline complet de détection de fraude sur des transactions par carte bancaire : un modèle de machine learning entraîné sur des données historiques est appliqué en temps réel à un flux de transactions (Kafka + Spark Streaming), les résultats sont stockés dans MongoDB et visualisés dans un dashboard Streamlit.

## Architecture

```
creditcard.csv ──► producer.py ──► Kafka (topic "transactions")
                                        │
                                        ▼
                     consumer.py (Spark Structured Streaming)
                     • recrée les features   • applique le modèle
                                        │
                                        ▼
                                    MongoDB
                     ┌──────────────────┴──────────────────┐
              toutes_transactions                   fraudes_detectees
                     │
                     ▼
              dashboard.py (Streamlit)
```

## Technologies

| Rôle | Outil |
|---|---|
| Modèle | scikit-learn (LogisticRegression), imbalanced-learn (SMOTE) |
| Streaming | Apache Kafka, Spark Structured Streaming (PySpark 3.4) |
| Stockage | MongoDB |
| Dashboard | Streamlit, Plotly |
| Langage | Python 3.12, Java 11 (requis par Spark) |

## Structure du projet

```
detection_fraude_project/
├── notebooks/bank_2.ipynb        # Exploration, feature engineering, comparaison des modèles
├── src/
│   ├── modele/modele.py          # Entraînement → fraud_model.pkl + scaler.pkl
│   ├── producer/producer.py      # Rejoue le dataset vers Kafka
│   └── consumer/consumer.py      # Spark : prédiction + sauvegarde MongoDB
├── dashboard.py                  # Dashboard Streamlit
├── creditcard.csv                # Dataset brut (non versionné)
└── README.md
```

## Données

Dataset Kaggle *Credit Card Fraud Detection* : 284 807 transactions, dont 492 fraudes (~0,17 %). Colonnes : `Time`, `Amount`, `V1`…`V28` (composantes PCA anonymisées) et `Class` (0 = normale, 1 = fraude).

## Le modèle

**Features finales (dans cet ordre exact)** : `V1…V28`, `hour_sin`, `hour_cos`, `Amount_standarise`.

- `Amount_log = log1p(Amount)`, puis standardisation (`StandardScaler`) → `Amount_standarise`
- `hour` déduite de `Time`, encodée de façon cyclique (`hour_sin`, `hour_cos`)
- Split 80/20 stratifié, **SMOTE appliqué uniquement sur le train** pour gérer le déséquilibre
- Modèle : `LogisticRegression`

> ⚠️ L'ordre des colonnes doit rester identique entre `modele.py` et `consumer.py` (`FEATURE_ORDER`), sinon les prédictions sont fausses sans qu'aucune erreur ne soit levée.

**Résultat observé sur le flux** (≈ 15 800 transactions rejouées, seuil 0,5) :

| | Fraude réelle | Non-fraude réelle |
|---|---|---|
| **Alerte** | 58 | 273 |
| **Pas d'alerte** | 3 | 15 502 |

Recall ≈ 95 %, précision ≈ 17 %. Le modèle détecte presque toutes les fraudes mais génère beaucoup de fausses alertes, ce qui est typique de SMOTE + régression logistique. Le curseur de seuil du dashboard permet d'arbitrer entre recall et précision sans réentraîner.

## Installation

Prérequis : Python 3.10+, Java 11, Kafka, MongoDB.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn joblib \
            kafka-python pyspark==3.4.0 pymongo pyarrow \
            streamlit plotly "pandas<3.0.0"
```

## Configuration à adapter

- `modele.py` : `CSV_PATH` (chemin vers `creditcard.csv`)
- `producer.py` : `CSV_PATH`, `DELAY_SECONDS` (vitesse du flux)
- `consumer.py` : `MODEL_PATH`, `SCALER_PATH`, `JAVA_HOME`
- Kafka : `127.0.0.1:9092` — MongoDB : `mongodb://localhost:27017`

## Lancement (dans cet ordre)

**1. Entraîner le modèle**
```bash
python3 src/modele/modele.py
```
Génère `fraud_model.pkl` et `scaler.pkl`.

**2. Démarrer Kafka et MongoDB**, puis créer le topic (une seule fois) :
```bash
kafka-topics.sh --create --topic transactions --bootstrap-server 127.0.0.1:9092
```

**3. Lancer le consumer** (terminal 1 — il doit être prêt avant le producer)
```bash
python3 src/consumer/consumer.py
```

**4. Lancer le producer** (terminal 2)
```bash
python3 src/producer/producer.py
```

**5. Lancer le dashboard** (terminal 3)
```bash
streamlit run dashboard.py
```
Ouvrir http://localhost:8501.

## Données dans MongoDB

Base `fraud_detection` :

- **`toutes_transactions`** : toutes les transactions traitées
- **`fraudes_detectees`** : uniquement celles prédites comme fraude

Exemple de document :
```json
{
  "transaction_id": "f371df8f-b94e-41eb-967c-0bbcce108265",
  "Time": 9087, "Amount": 6.99,
  "hour_sin": 0.5, "hour_cos": 0.866,
  "prediction": 0, "proba_fraude": 0.0133, "Class": 0
}
```
`Class` est la vérité terrain, conservée uniquement pour évaluer le système, jamais utilisée pour prédire.

Vérifier la qualité du modèle sur le flux (matrice de confusion) :
```js
db.toutes_transactions.aggregate([
  { $group: { _id: { prediction: "$prediction", Class: "$Class" }, n: { $sum: 1 } } }
])
```

## Dashboard

- **Tableau de bord** : indicateurs clés (transactions, alertes, montant alerté, recall, précision), dernières alertes, matrice de confusion
- **Alertes par heure** : alertes par heure de la journée et évolution dans le temps
- **Distributions** : montants et probabilités de fraude
- **Barre latérale** : seuil de décision, montant minimum, rafraîchissement automatique

## Problèmes fréquents

| Symptôme | Cause / solution |
|---|---|
| `SyntaxError: unterminated string literal` | Guillemets dupliqués dans un chemin (`CSV_PATH`) |
| Warning PyArrow / pandas ≥ 3.0 | Sans gravité ; `pip install pyarrow "pandas<3.0.0"` pour le supprimer |
| Collection vide dans MongoDB | Vérifier le nom (`toutes_transactions`) et écrire `countDocuments({})` avec ses accolades |
| `fraudes_detectees` absente | Créée au premier document inséré : elle apparaît dès la première fraude détectée |
| Aucun message reçu par Spark | Lancer le consumer avant le producer ; vérifier que le topic existe |

## Limites et améliorations possibles

- Précision faible (~17 %) : ajuster le seuil, ou tester XGBoost (déjà comparé dans le notebook)
- Évaluer avec la PR-AUC plutôt que l'accuracy, trompeuse sur des classes aussi déséquilibrées
- Le flux est une simulation (rejeu du CSV) : les features `V1…V28` sont propres à ce dataset
- Ajouter un `checkpointLocation` au stream Spark pour la reprise après arrêt
- Le dashboard lit toute la collection : passer à des agrégations MongoDB si le volume grossit
