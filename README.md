
# 🌟 Rakamin HR Intelligence Hub  
*A data-powered talent intelligence system built to reveal performance, potential, and promotion readiness.*

---

<img width="1774" height="997" alt="The Promotion Paradox" src="https://github.com/user-attachments/assets/7501b515-10a6-47d3-a366-8e86519a2554" />

## 📖 Story Behind the Project  

In many modern organizations, employee data lives in silos: spreadsheets scattered across teams, inconsistent performance reviews, and talent decisions that rely more on intuition than insight.  
We wondered: **can a company understand the “heartbeat” of its workforce in real time?**

That question sparked the creation of **Rakamin HR Intelligence Hub**—  
a centralized intelligence system designed to transform raw HR data into meaningful insight, narrative, and prediction.

The system combines **data engineering**, **unsupervised learning**, **supervised modeling**, and **intuitive dashboards** to help decision‑makers understand talents not just as records, but as evolving stories.

---

## 🎯 What This Platform Solves  

- HR teams struggle to see macro & micro patterns in talent data.  
- Identifying high performers, at‑risk talent, and future leaders is often subjective.  
- Organizations lack a transparent, data-driven way to identify promotion‑ready individuals.  
- Talent decisions rarely integrate *performance*, *leadership*, and *potential* in one framework.

**Rakamin HR Intelligence Hub** addresses all of these through a fully interactive analytics and ML-powered prediction system.

---

## 🧠 System Architecture

### 1️⃣ Data Foundation  
The journey begins with a cleaned dataset containing performance metrics, leadership scores, training exposure, project involvement, and peer reviews.

From this foundation, we construct three critical indexes:

- **Performance Index**  
- **Leadership Index**  
- **Potential Index**  

These indexes act as the “DNA” that shapes everything the system knows about a talent.

---

<img width="803" height="363" alt="Best model is K-Means with K = 4" src="https://github.com/user-attachments/assets/51d41c5b-a4f7-4f1b-933d-0f3e3f63e667" />

### 2️⃣ Talent Clustering: Discovering Personas  
Using **K-Means Clustering**, the system identifies four natural groups of talent:

1. *Underdeveloped with Potential*  
2. *At-Risk and Underpowered*  
3. *All-Around Top Performer*  
4. *Consistent Performer / Emerging Leader*

Each cluster is enriched with:

- Characteristics  
- Behavioral interpretation  
- HR recommendations  
- Suitable development programs  

It transforms numbers into *personas*.

---

### 3️⃣ Promotion Eligibility Model  
A curated **Logistic Regression** model analyzes 9 key indicators:

- Performance Index  
- Potential Index  
- Leadership Index  
- Project involvement  
- Growth trajectory  
- Consistency metrics  
- And more...

The model generates:

- **Promotion Score**
- **Eligible / Not Eligible**
- **Explanation of the result**
- **Radar chart visualizing strengths and gaps**
- **Succession potential category**

It doesn't just say *yes* or *no*—  
it tells *why*.

---

### 4️⃣ Streamlit Application  

<img width="1887" height="1102" alt="Dashboard prototype" src="https://github.com/user-attachments/assets/41f9282b-bbaa-4f92-8d8c-d5fba3a0d042" />

Key modules inside the app:

#### **1. Talent Overview**
A macro-level dashboard revealing:

- Workforce distribution  
- Age demographics  
- Position levels  
- Performance group counts  
- Potential loss estimation  

#### **2. Talent Performance**
Dive deeper into:

- Top talent ranking  
- High-risk employee lists  
- Index comparisons  
- Leadership vs potential patterns  

#### **3. Talent Predictor**
The most dynamic module:

- Select existing employee  
- Predict new employee manually  
- Bulk upload CSV with instant enrichment  
- Automatic clustering  
- Machine learning predictions  
- Narrative explanation + development suggestions  

This module brings analytics and AI together in a single experience.

---

## 🚀 Live Demo  
Experience the system here:  
👉 **https://rakaminhrdashboard.streamlit.app/**

<img width="1878" height="1209" alt="Prediction system" src="https://github.com/user-attachments/assets/2799734a-16d7-4b99-aa7b-6426e25dd152" />

---

## 📂 Repository Structure  

```
syntaxsociety/
│
├── app.py                       # Streamlit application
├── data/
│   ├── Clean/                   # Cleaned dataset used for dashboard
│   └── Raw/                     # Raw input data (optional)
│
├── models/
│   ├── cluster_model.pkl        # KMeans clustering model
│   ├── cluster_scaler.pkl       # Scaler for feature engineering
│   ├── cluster_scaler_3.pkl     # Scaler for 3-index clustering
│   ├── cluster_metadata.json    # Cluster interpretation
│   └── logistic_pipeline.pkl    # Promotion eligibility model
│
├── scripts/
│   └── train_eligibility.py     # End-to-end model training pipeline
│
├── notebooks/                   # Exploratory analysis & experimentation
│
└── reports/
    └── Syntax Society Project Report.pdf
```

---

## 🧪 Machine Learning Development  

The training pipeline (located in `/scripts/train_eligibility.py`) includes:

- Full feature engineering  
- Clustering model generation  
- Model persistence  
- Promotion eligibility model training  
- GridSearchCV optimization  
- Handling class imbalance via SMOTETomek  
- Exporting final models & metadata  

This gave the system:

- **Stable cluster separation**  
- **High interpretability**  
- **Trustworthy promotion predictions**  

---

## 🧩 Key Design Principles  

- **Explainability first** — ML predictions always come with narrative reasoning  
- **Consistency** — standardized index formulas across system  
- **Scalability** — supports bulk upload & dynamic enrichment  
- **Modularity** — training pipeline decoupled from Streamlit UI  
- **Human-friendly UI** — clean, colorful, and narrative-driven  

---

## 🛠️ How to Run Locally  

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure model files exist under `/models`.  
If not, re-run the training pipeline:

```bash
python scripts/train_eligibility.py
```

---

## 🌱 Future Enhancements  

- Role-based dashboards for HRBP, Manager, and C‑level  
- Predictive attrition modeling  
- Team-level organizational health metrics  
- Skill gap analysis for workforce planning  
- Integration with HRIS or ATS platforms  

---

## 👤 Authors  
**Syntax Society**  

---

## 🙏 Closing Note  

This project was created not only as a dashboard,  
but as a **storytelling engine for human potential**.  

Data reveals patterns.  
Models reveal probabilities.  
But insights?  
Insights reveal *people*. Their strengths, struggles, and opportunities to grow.

The Rakamin HR Intelligence Hub is built with that philosophy at its core.

---
