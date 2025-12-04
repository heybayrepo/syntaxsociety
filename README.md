# 📘 **Rakamin HR Intelligence Hub**  
**Data-Driven Workforce Insights, Talent Mapping & Promotion Prediction**

Live Dashboard → **https://rakaminhrdashboard.streamlit.app/**  
Repository → **https://github.com/heybayrepo/syntaxsociety**

---

## ⭐ Overview
Rakamin HR Intelligence Hub is an interactive, end-to-end analytics system to help HR teams evaluate employee performance, leadership, potential, and promotion readiness using a clean UI and an interpretable machine-learning model.

The project includes:
- Cleaned & engineered HR dataset  
- Talent clustering for segmentation  
- Logistic Regression promotion eligibility model  
- Full Streamlit dashboard for exploration, scoring, and recommendations  
- A final project report detailing methodology, metrics, and monitoring strategy  

📄 **Full report:**  
`/report/Syntax Society - Project Report.pdf`

---

## 🖼 Dashboard Preview  
(Add your images inside `docs/images/` and replace these placeholders)

![Dashboard Overview](docs/images/hero-dashboard.png)  
![Talent Overview Tab](docs/images/tab-overview.png)  
![Talent Performance Tab](docs/images/tab-performance.png)  
![Talent Predictor Tab](docs/images/tab-predictor.png)

---

## 🎯 Key Features

### 🔍 1. Talent Overview
- Workforce summary  
- Performance groups  
- Outlier detection & potential salary loss  

### 📊 2. Talent Performance
- Ranking by performance, leadership, potential  
- Level-based filtering  

### 🧠 3. Talent Predictor (ML Model)
- Promotion eligibility prediction  
- Strengths & weaknesses  
- Radar chart visualization  
- Succession potential scoring  
- Supports ID selection, manual input, and CSV upload  

### 🧩 Clustering
K-Means clustering on:  
- Performance Index  
- Leadership Index  
- Potential Index  

---

## 🧬 Data & Model Summary

### Data Preparation
- Missing value handling  
- Outlier removal  
- Feature engineering  
- Final indexes used for modeling  

### ML Model
- Logistic Regression  
- SMOTETomek balancing  
- GridSearchCV tuning  
- Metrics: F1, ROC-AUC, Precision, Recall  

---

## 🧱 System Architecture

(Add architecture image: `docs/images/pipeline-arch.png`)

---

## 🚀 Deployment

### Local Setup
```
git clone https://github.com/heybayrepo/syntaxsociety
cd syntaxsociety
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
Auto-deploys from `main`.

---

## 📂 Repository Structure

```
/
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Clean/
│       └── dataset_clustered_dashboard.csv
│
├── models/
│   └── logistic_pipeline.pkl
│
├── docs/
│   └── images/
│
├── report/
│   └── Syntax Society - Project Report.pdf
│
└── notebooks/
    ├── 01_EDA.ipynb
    ├── 02_Feature_Engineering.ipynb
    └── 03_Modeling.ipynb
```

---

## 🧪 Monitoring
- Drift detection  
- Model recalibration  
- Cluster stability checks  

---

## 📩 Contributing
PRs welcome.

---

## 👤 Maintainer
Bayu / Syntax Society  
https://github.com/heybayrepo
