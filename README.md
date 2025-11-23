# **The Promotion Paradox: Fixing Biased Talent Promotion Decisions Through Data-Driven HR Analytics**

## **By:** Syntax Society

<img width="600" height="338" alt="Screenshot 2025-11-21 at 14 41 40" src="https://github.com/user-attachments/assets/35e726c9-bfb6-49f1-9517-0fdec9c2a176" />

## **Team:**
- Bayu Maitra (Project Manager)
- Dian Ulhaq (Data Analyst)
- Maslahatul Husna (Data Engineer)
- Febiansyah Ahnaf (Data Scientist)
- Keisya Nisrani (Data Scientist)

## **Repository:** 

syntaxsociety

## **Repo structure:**

```bash
📁 syntaxsociety/
├── notebooks/       → Exploratory Data Analysis (EDA) & modeling
├── assets/          → Images
├── data/            → Raw datasets (stored locally / Google Drive)
├── reports/         → Reports, charts, and final deliverables
└── README.md        → Project documentation & collaboration guide
```

## **Clone the repository**
- *git clone https://github.com/heybayrepo/syntaxsociety.git*
- *cd syntaxsociety*

## **Project Overview**

<img width="600" height="338" alt="Screenshot 2025-11-22 at 12 02 47" src="https://github.com/user-attachments/assets/46066438-0428-4aba-8b7f-5066100a0ca9" />

This project focuses on predicting employee promotion eligibility using structured HR data such as performance scores, leadership metrics, peer review scores, project experience, and training hours.

The motivation comes from a clear gap identified in the HR workflow:
- Manual data entry → numerous inconsistencies and anomalies
- Subjective judgments → biased promotion decisions
- Lack of standardized, explainable evaluation tools

Through data cleaning, feature engineering, and machine learning modeling, this project aims to support HR with a fair, objective, transparent decision-support system.

<img width="600" height="338" alt="Screenshot 2025-11-22 at 12 05 42" src="https://github.com/user-attachments/assets/d9299cbe-8a33-4901-83c0-de12236aa737" />

## **The main objectives are to:**

	1.	Develop and evaluate multiple ML models for promotion prediction.
	2.	Engineer new features that better capture employee performance and potential.
	3.	Deliver a transparent, fair, and explainable model to support HR decision-making.

## **Dataset Summary**

<img width="600" height="338" alt="Screenshot 2025-11-22 at 12 08 30" src="https://github.com/user-attachments/assets/8c183a11-99f3-433f-a926-b288196faa66" />

Talent Promotion Dataset provided by Rakamin Academy.
Dataset is stored in this repository.

**Structure**
- Rows: 1000
- Columns: 10
- Target Variable: Promotion_Eligible
	0 → Not eligible
	1 → Eligible

**Key Issues Identified During EDA**
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

1. Outlier Handling
	- Removed ~1.3% data entries with extreme values (Age, Training Hours).
	- Dropped Years_at_Company due to consistent logical errors.

3. Missing Value Treatment
	- Numeric → Median
 	- Categorical → Mode
  	- Result: 0 missing values.

<img width="600" height="338" alt="Screenshot 2025-11-22 at 12 10 59" src="https://github.com/user-attachments/assets/b568aef9-dc89-40c5-8a4d-1ed052c0cd36" />

4. Feature Selection

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

5. Feature Engineering

Two new features were created to reduce bias and improve representational power:
- Leadership_Index = (Leadership Score + Peer Review Score) / 2
- Performance_Index = (Performance Score + Projects Handled + Training Hours) / 3
- Potential_Index = (0.4 x Training_Hours_scaled) + (0.4 x Peer_Review_Score_scaled) + (0.2 x Leadership_Score_scaled)
- Growth_Momentum = Projects_Handled_scaled / (Training_Hours_scaled + 1)
- Leadership_Influence = Peer_Review_Score_scaled / (Leadership_Score_scaled + 1)
- Performance_Consistency = Performance_Score x Projects_Handled_scaled


6. Standardization & Balancing
	•	All numeric features were standardized.
	•	Target kept as binary.
	•	SMOTE applied to balance the dataset.

<img width="600" height="338" alt="Screenshot 2025-11-22 at 12 16 22" src="https://github.com/user-attachments/assets/6dd10242-86b8-4f4b-a1af-b227244781d8" />

