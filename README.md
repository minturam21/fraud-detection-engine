# 🛡️ **Fraud Risk Intelligence System**

*A Production-Style Fraud Detection Pipeline with Rules + Machine Learning*

---

## 🔍 **1. Overview**

This project implements a **Fraud Risk Intelligence System** inspired by real-world financial fraud engines used in modern fintech companies.
It combines:

* **Behavioral rule engine**
* **Machine learning fraud model**
* **Risk score blending**
* **Threshold-based decisioning**
* **Event-level feature engineering**
* **Temporal validation**
* **Synthetic dataset generation**
* **Logging & monitoring components**

The system ingests raw user activity events (logins, password resets, transactions), converts them into behavioral features, and produces a final decision:

```
ALLOW / OTP_CHALLENGE / BLOCK
```

This project is fully modular, production-oriented, and suitable for real-time integration.

---

## 🏗️ **2. System Architecture**

```mermaid
flowchart TD

    A[Raw Events<br>(login, reset, txn)] --> B[Data Cleaning<br>- missing values<br>- invalid timestamps<br>- invalid amounts]

    B --> C[Feature Engineering<br>- velocity<br>- amount deviation<br>- device/IP intelligence<br>- location jump]

    C --> D[Rule Engine<br>- brute force login<br>- new device<br>- first-time receiver<br>- instant reset<br><b>Outputs:</b> rule_score & rule_flags]

    C --> E[ML Fraud Model<br>RandomForest<br><b>Outputs:</b> model_score]

    D --> F[Score Combiner<br>final_score = α*model + (1-α)*rule]
    E --> F

    F --> G[Decision Engine<br>- thresholds.json<br>- ALLOW / OTP / BLOCK]

    G --> H[Logging & Monitoring<br>fraud_engine.log]

```

---

## 🧪 **3. Dataset (Synthetic)**

Since real financial fraud data is confidential and regulated, this project uses a **synthetic dataset** that simulates:

* 10,000 events
* 500 users
* realistic transaction patterns
* login failures
* device & IP changes
* high-amount anomalies
* location changes
* engineered fraud labels

Generated using:

```
data/synthetic/generate_synthetic_data.py
```

Output file:

```
data/synthetic/transactions.csv
```

---

## ⚙️ **4. End-to-End Pipeline**

The training pipeline runs in the following sequence:

1. **Load synthetic raw data**
2. **Clean invalid rows (missing values, impossible sequences)**
3. **Feature engineering (behavior + velocity features)**
4. **Temporal train/test split** (prevents leakage)
5. **Handle class imbalance** (fraud is rare)
6. **Train ML model** (Random Forest)
7. **Evaluate with professional metrics**
8. **Generate thresholds**
9. **Save model + metadata**

Pipeline script:

```
python pipeline/run_training_pipeline.py
```

---

## 🧠 **5. Feature Engineering**

Behavioral and statistical features include:

### **Login & Device Features**

* `failed_login_10min`
* `new_device_flag`
* `new_ip_flag`

### **Transaction Features**

* `z_amount` (deviation from user’s typical amounts)
* `tx_count_5min`
* `first_time_receiver_flag`

### **Location Features**

* `distance_from_last_location_km`

### **Security Features**

* `time_between_login_reset_sec`

These features replicate signals used in real fraud surveillance systems.

---

## 🧩 **6. Rule Engine**

A lightweight but high-impact rule engine assigns:

```
rule_score (0–1)
rule_flags (list of triggered rules)
```

Rules include:

* Too many failed logins (brute-force takeover)
* New device used by user
* First-time money receiver
* Instant password reset after login
* High-risk location jump
* Unusual transaction amount

File:

```
scoring/rule_engine.py
```

---

## 🤖 **7. Machine Learning Model**

A **RandomForestClassifier** is trained using:

* temporal split
* balanced fraud data
* engineered behavioral features

Evaluation metrics:

* **AUC**
* **Recall @ 1% False Positive Rate**
* **Precision @ K**
* **Fraud Capture Rate**
* **Detection Latency**

Evaluation script:

```
pipeline/evaluate.py
```

---

## 📊 **8. Threshold Logic**

Final decisions are made using `thresholds.json`:

```
{
  "low": 0.30,
  "medium": 0.55,
  "high": 0.80
}
```

Logic:

* `score >= high` → **BLOCK**
* `score >= medium` → **OTP Challenge**
* `score < medium` → **ALLOW**

---

## 🧮 **9. Decision Pipeline**

This component merges:

* rule engine
* ML model score
* weighted score combiner
* threshold logic

Returns a structured response:

```
{
  "action": "BLOCK",
  "final_score": 0.82,
  "reasons": ["score>=high", "rule:new_device"],
  "context": {"user_id": 101, "amount": 7000}
}
```

File:

```
scoring/decision_pipeline.py
```

---

## 📈 **10. Monitoring & Logging**

A production-style logger tracks:

* rule triggers
* model scores
* final decisions
* training events
* errors

Logs saved in:

```
logs/fraud_engine.log
```

---

## 📉 **11. Visual Results**

Generated using:

```
python pipeline/make_plots.py
```

Visuals stored in:

```
data/plots/
```

### Included Plots

* Rolling Fraud Rate Over Time
* Transaction Amount Distribution
* Fraud vs Legit Amount Comparison
* Device Usage Patterns
* Failed Login Velocity Distribution

These plots are included in the README or uploaded to GitHub for stakeholders.

---

## ▶️ **12. How to Run the System**

### **Install Dependencies**

```
pip install -r requirements.txt
```

### **Generate Synthetic Data**

```
python data/synthetic/generate_synthetic_data.py
```

### **Run Training Pipeline**

```
python pipeline/run_training_pipeline.py
```

### **Run Full System Test**

```
python tests/run_full_system_test.py
```

---

## 📁 **13. Project Structure**

```
├── data
│   ├── synthetic
│   │   ├── transactions.csv
│   │   └── generate_synthetic_data.py
│   ├── models
│   │   ├── model.pkl
│   │   ├── thresholds.json
│   │   └── metadata.json
│   └── plots
│       ├── amount_distribution.png
│       ├── fraud_rate_over_time.png
│       └── ...
│
├── pipeline
│   ├── clean.py
│   ├── features.py
│   ├── split.py
│   ├── imbalance.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── make_plots.py
│   └── run_training_pipeline.py
│
├── scoring
│   ├── rule_engine.py
│   ├── score.py
│   └── decision_pipeline.py
│
├── utils
│   ├── loader.py
│   ├── validators.py
│   └── logger.py
│
├── tests
│   └── run_full_system_test.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 **14. Future Improvements**

Potential extensions include:

* FastAPI real-time scoring service
* Device fingerprint intelligence
* Advanced geolocation model
* Gradient Boosting ML models (XGBoost/LightGBM)
* Adaptive thresholds
* Fraud drift detection
* SHAP explainability reports
* Model retraining scheduler

---

## 👤 **15. Author**

Fraud Risk Intelligence System
Built by **Mintu Ramchiary**
For ML Engineer / Data Scientist roles.

---
