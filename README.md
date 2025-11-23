# **The Promotion Paradox: Fixing biased talent promotion decisions through data-driven HR analytics**

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

## **Model train and test**

### **Tree-based models**

This section evaluates the performance of three tree-based classification models, which is Decision Tree, Random Forest, and XGBoost, using an 80:20 train–test split. Initial results showed clear overfitting across baseline models, requiring tuning to improve generalization.

<img width="600" height="388" alt="Correlation, split data and target check" src="https://github.com/user-attachments/assets/05e99ca3-3b73-4da8-b9c5-705e39c67e80" />

#### Baseline model
**Decision Tree (default):** Achieved 100% accuracy on training data but low test performance (F1 32%, ROC-AUC 52%), indicating strong overfitting. This is our baseline model.
**Random Forest & XGBoost (default):** Both models also showed overfitting with perfect train scores but weak generalization on the test set.

#### Tuned model

- **Tuned Decision Tree:** Overfitting was reduced, but overall performance remained low. F1 (Test): 17% | Recall: 12%
- **Tuned Random Forest:** Demonstrated the best generalization and overall balance across metrics. F1 (Test): 36% | Recall: 45% | ROC-AUC: 54%
- **Tuned XGBoost:** Tuning did not improve performance; still underperformed. F1 (Test): 17% | Recall: 12%

#### Other model

- **Gaussian Naive Bayes:** Weak performance and very low F1 scores.
- **Neural Network (shallow):** Modest accuracy but still affected by underfitting and low Recall/F1.

<img width="600" height="338" alt="Best classification model" src="https://github.com/user-attachments/assets/07657a5d-e68a-425d-853e-491dc25e2f35" />

#### Conclusion

The tuned Random Forest achieved the strongest performance among all tested models, offering the best balance of F1, recall, and ROC-AUC on the test set. It stands as the most reliable model for predicting promotion eligibility at this stage.

### **Linear models**

This experiment evaluates two linear classification algorithms, Logistic Regression and Support Vector Machine (SVM), using a dataset where class imbalance was first addressed through SMOTE. Because linear models are sensitive to multicollinearity, highly correlated features were removed prior to training.

#### Preprocessing for linear models
- **Feature Selection:** Removed highly correlated variables (Performance_Score, Training_Hours, Projects_Handled, Peer_Review_Score) to avoid multicollinearity issues common in linear models.
- **Train–Test Split:** 80% training, 20% testing.
- **Class Balancing:** Applied SMOTE, balancing Eligible vs. Not Eligible from 221:568 → 568:568 in the training set.

#### Model performance

- **Logistic Regression (Default):** Achieved modest performance with F1 and Recall around 32%, and ROC-AUC 52%. Cross-validation helped stabilize results but overall predictive power remained limited.
- **Logistic Regression (Tuned):** Hyperparameter tuning slightly improved recall (33%) but did not meaningfully improve F1 or ROC-AUC (52%). Performance remained similar to the baseline.
- **SVM (Default):** Delivered the strongest results among all linear models, achieving: F1 (Test): 60%, Recall (Test): 60%, ROC-AUC (Test): 57%. This model demonstrated the best generalization and the most balanced predictive behavior.
- **SVM (Tuned):** Tuning did not outperform the default model, reducing performance to: F1 (Test): 47%, Recall (Test): 56%, ROC-AUC (Test): 53%. Default SVM remained more reliable.

<img width="600" height="338" alt="Best linear model" src="https://github.com/user-attachments/assets/9d21a52b-66b7-47de-8bf7-89716c0c0907" />

#### Conclusion

Across all linear-model experiments, the default SVM model provided the strongest and most stable results. Logistic Regression showed limited predictive strength, while SVM consistently demonstrated better recall, F1-score, and ROC-AUC, making it the most dependable linear approach for this dataset.

### **Unsupervised learning**

This section explores clustering methods to uncover natural groupings of employee performance profiles. The analysis was conducted on two datasets: one using all available features and another using a selected subset after removing highly correlated variables.

#### Feature selection

To reduce noise and avoid distortion in cluster formation, several highly correlated features were removed for the reduced-feature experiment:
- Performance_Index
- Training_Hours
Two datasets were then tested: full features vs. selected features.

#### Clustering models used

Several unsupervised algorithms were evaluated to identify the most meaningful employee segments:
- K-Means
- Gaussian Mixture Model (GMM)
- K-Medoids

#### Optimal K selection

The optimal number of clusters was determined using the Elbow Method, revealing that K = 5 produced the clearest and most stable segmentation structure.

<img width="600" height="388" alt="Unsupervised models evaluation" src="https://github.com/user-attachments/assets/fe83640f-5157-4628-96f0-109f29de4d54" />

#### Model Evaluation
- **K-Means:** Achieved the highest overall Silhouette Score across K values. Produced consistently lower DBI scores, indicating better-defined and more compact clusters. At K = 5, K-Means showed the strongest balance between cluster separation and cohesion, leading to stable and meaningful cluster structures.
- **Gaussian Mixture Model (GMM):** Demonstrated lower Silhouette Scores than K-Means across nearly all K values. DBI scores were significantly higher, suggesting loose, less distinct clusters. Cluster boundaries tended to overlap more due to the probabilistic nature of GMM, reducing interpretability.
- **K-Medoids:** Performed better than GMM but still slightly weaker than K-Means. Silhouette Scores were moderate but did not surpass K-Means for any K. DBI values were acceptable but consistently higher than K-Means, indicating less compact clusters.

#### Conclusion

Across all tested algorithms, K-Means with K = 5 delivered the most coherent, interpretable, and well-separated clusters. This makes it the most suitable unsupervised method for segmenting employee performance profiles in this dataset.

#### Cluster Interpretation

Using PCA visualizations and radar charts, the best-performing K-Means model (K = 5) revealed five distinct employee segments, each representing different talent profiles:
- **Cluster 0 — Emerging Performers:** Strong technical capability but still require leadership development.
- **Cluster 1 — Leadership-Oriented:** Employees with good leadership presence but needing improvement in execution skills.
- **Cluster 2 — Low Consistent Performers:** Require coaching, targeted training, or potential role reassessment.
- **Cluster 3 — Reliable Core Talent:** Reliable, consistent performers who maintain day-to-day output.
- **Cluster 4 — Strategic High Performers:** Unique high-performing individuals with specialty strengths; ideal for strategic roles.

<img width="600" height="388" alt="Modeling results and summary" src="https://github.com/user-attachments/assets/1c2e9238-cf92-469a-bf44-3011fb688381" />

## **Model selection summary**

Across all modeling experiments — tree-based models, linear models, and unsupervised clustering — the dataset showed patterns that were more continuous than sharply separable. As a result, most supervised models struggled to produce reliable predictions, even after tuning.

Based on all experiment, the most suitable model for this project is K-Means with K = 5. The reasons are:
1. It produced the most stable, most compact, and most interpretable cluster structure.
2. The resulting 5 clusters align naturally with HR personas, such as emerging high performers, leadership-strong profiles, consistent low performers, stable operators, and specialized high-outliers.
3. Metrics showed K-Means was more reliable than K-Medoids and significantly better than GMM.
4. Unlike supervised models — which all struggled due to weak promotion-eligibility signals — clustering captured continuous performance patterns more effectively.

## **Practical value for HR**

The five clusters uncovered through K-Means don’t just describe statistical groupings, they map directly onto real talent personas inside the organization. Each segment helps HR see the workforce with new clarity. Instead of relying on intuition or fragmented data, HR can finally understand who needs what and why.

With these clusters, HR can:
- **Plan succession pipelines more confidently:** Emerging high performers and leadership-ready profiles can be identified early, allowing HR to prepare them for future roles instead of waiting for annual reviews or manager referrals.
- **Design targeted development programs:** Instead of one-size-fits-all training, HR can tailor interventions: coaching for consistent low performers, leadership development for Cluster 0 and 1, or strategic project placement for high-outlier talent.
- **Make more informed placement and staffing decisions:** Teams can be balanced more intentionally by distributing strengths and mitigating weaknesses based on cluster characteristics.
- **Spot high-potential and at-risk employees before issues surface:** Clusters reveal patterns that often remain hidden: who is quietly excelling, who is stagnating, and who may require closer support, all without relying solely on subjective manager evaluations.



















