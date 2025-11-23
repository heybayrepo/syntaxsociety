# **The Promotion Paradox: Fixing Biased Talent Promotion Decisions Through Data-Driven HR Analytics**

## **By:** Syntax Society

<img width="600" height="338" alt="The Promotion Paradox" src="https://github.com/user-attachments/assets/35e726c9-bfb6-49f1-9517-0fdec9c2a176" />

## **Team:**
- Bayu Maitra (Project Manager)
- Dian Ulhaq (Data Analyst)
- Maslahatul Husna (Data Engineer)
- Febiansyah Ahnaf (Data Scientist)
- Keisya Nisrani (Data Scientist)

## **Project overview**

This project focuses on predicting employee promotion eligibility using structured HR data such as performance scores, leadership metrics, peer review scores, project experience, and training hours. Through data cleaning, feature engineering, and machine learning modeling, this project aims to support HR with a fair, objective, transparent decision-support system.

<img width="600" height="338" alt="HR talent industry overview" src="https://github.com/user-attachments/assets/46066438-0428-4aba-8b7f-5066100a0ca9" />

## **Business understanding**

Promotion decisions play a critical role in shaping employee motivation, retention, and long-term organizational health. When the promotion process is subjective or inconsistent, high-potential employees may be overlooked, leading to frustration, disengagement, and increased turnover costs.

In many organizations, including the context represented by this dataset, HR teams still rely heavily on manual data entry, peer impressions, and loosely defined criteria. These gaps make it difficult to identify talent objectively and fairly. A data-driven approach is needed to support HR in building a transparent, consistent, and equitable promotion framework.

## **Problem statement**

Exploratory analysis of the Talent Promotion dataset reveals several issues that directly undermine fair and accurate promotion decisions:

- Data quality problems, including missing values, implausible ages, unrealistic training records, and inconsistent work-tenure information.
- Bias in promotion criteria, where decisions appear heavily influenced by only a few features such as peer review score or project count.
- Lack of standardization, making it difficult to compare employees on equal footing.
- Imbalanced promotion outcomes, with only 29% of employees marked as eligible in the raw data.

These challenges prevent HR from reliably identifying employees with genuine potential for advancement. A robust machine learning model is required to clean the data, learn meaningful performance patterns, and offer a more objective prediction of promotion eligibility.

<img width="600" height="338" alt="Gaps that need to be addressed" src="https://github.com/user-attachments/assets/d9299cbe-8a33-4901-83c0-de12236aa737" />

## **The main objectives are to:**

1. Develop and evaluate multiple ML models for promotion prediction.
2. Engineer new features that better capture employee performance and potential.
3. Deliver a transparent, fair, and explainable model to support HR decision-making.

## **Repository:** 

syntaxsociety

### **Repo structure:**

```bash
📁 syntaxsociety/
├── notebooks/       → Exploratory Data Analysis (EDA) & modeling
├── assets/          → Images
├── data/            → Raw datasets (stored locally / Google Drive)
├── reports/         → Reports, charts, and final deliverables
└── README.md        → Project documentation & collaboration guide
```

### **Clone the repository**
- *git clone https://github.com/heybayrepo/syntaxsociety.git*
- *cd syntaxsociety*

## **Dataset Summary**

<img width="600" height="338" alt="Rakamin's data characteristic" src="https://github.com/user-attachments/assets/8c183a11-99f3-433f-a926-b288196faa66" />

Talent Promotion Dataset provided by Rakamin Academy.
Dataset is stored in this repository.

### Structure
- Rows: 1000
- Columns: 10
- Target Variable: Promotion_Eligible
	0 → Not eligible
	1 → Eligible

### Key Issues Identified During EDA
- 449 missing values
- Implausible values:
	Negative age
	Employees under 18 years
	Training hours up to 5000
	Years_at_Company > Age
	Target imbalance: 29% eligible vs 71% not eligible
	Outliers in Age, Training_Hours, and Years_at_Company

These issues were addressed through imputation, outlier removal, and removal of unreliable columns.

## **Data Cleaning & Preprocessing**

### Outlier Handling
- Removed ~1.3% data entries with extreme values (Age, Training Hours).
- Dropped Years_at_Company due to consistent logical errors.

### Missing Value Treatment
- Numeric → Median
- Categorical → Mode
- Result: 0 missing values

<img width="600" height="338" alt="Feature selection strategy" src="https://github.com/user-attachments/assets/b568aef9-dc89-40c5-8a4d-1ed052c0cd36" />

### Feature Selection

Dropped:
- Employee_ID
- Age
- Years_at_Company
- Current_Position_Level

Used:
- Performance_Score
- Leadership_Score
- Peer_Review_Score
- Projects_Handled
- Training_Hours
- Promotion_Eligible

### Feature Engineering

Two new features were created to reduce bias and improve representational power:
- Leadership_Index = (Leadership Score + Peer Review Score) / 2
- Performance_Index = (Performance Score + Projects Handled + Training Hours) / 3
- Potential_Index = (0.4 x Training_Hours_scaled) + (0.4 x Peer_Review_Score_scaled) + (0.2 x Leadership_Score_scaled)
- Growth_Momentum = Projects_Handled_scaled / (Training_Hours_scaled + 1)
- Leadership_Influence = Peer_Review_Score_scaled / (Leadership_Score_scaled + 1)
- Performance_Consistency = Performance_Score x Projects_Handled_scaled

### Standardization & Balancing
- All numeric features were standardized
- Target kept as binary
- SMOTE applied to balance the dataset

<img width="600" height="338" alt="Data cleansing and correlation map" src="https://github.com/user-attachments/assets/6dd10242-86b8-4f4b-a1af-b227244781d8" />

## **Evaluation metrics**

Before evaluating the machine learning models, it’s important to define how model performance is measured. This project uses two different categories of metrics depending on the type of method: unsupervised clustering and supervised classification.

<img width="600" height="338" alt="Evaluation metrics" src="https://github.com/user-attachments/assets/ceb06341-0736-4f7f-9f2b-c5e8c443fae1" />

### Unsupervised Evaluation (clustering)

- **Silhouette Score:** Measures how well each employee fits within their assigned cluster compared to other clusters. This is useful 	because human performance data tends to be continuous and overlapping; silhouette helps reveal whether the 		model is forcing unnatural separations.
- **Davies–Bouldin Index (DBI):** A complementary metric that evaluates intra-cluster similarity versus inter-cluster separation. DBI is 			sensitive to cluster overlap — a common situation in HR performance — and often exposes hidden structure that 	silhouette alone cannot capture.

### Supervised Evaluation (tree-based and linear models)

- **ROC-AUC:** Measures the model’s ability to distinguish between eligible and not eligible employees across thresholds. 		ROC-AUC is stable under class imbalance and allows fair comparison across tree-based and linear models.
- **F1-Score:** Since the target is imbalanced, F1 provides a balanced measure of precision and recall. It prevents the model from achieving high accuracy by simply predicting the majority class.
- **Confusion Matrix:** HR needs visibility into false positives (promoting the wrong person) and false negatives (overlooking eligible talent). The confusion matrix shows these error types clearly and supports transparent HR decision-making.
- **Recall:** Missing eligible employees (false negatives) is costlier for HR than flagging someone incorrectly. High recall ensures the model captures as many genuinely promotable employees as possible.
- **Precision:** Precision ensures the model’s “eligible” predictions are trustworthy. High precision reduces wasted time and resources on reviewing or promoting employees who are not truly ready.




























