# Fraud Detection ML Project

A complete machine learning pipeline to detect fraudulent financial transactions using Random Forest.  
Achieved **99.97% accuracy** and **0.88 F1-score** on real-world, imbalanced data.

---

##  Project Overview

-  Cleaned and analyzed a **6.3 million+ row** dataset
-  Detected rare frauds using RandomForestClassifier
-  Addressed extreme class imbalance with SMOTE
-  Achieved **1.00 precision**, **0.79 recall**, **0.88 F1-score**
-  Interpreted business logic, flagged fraud indicators, and suggested preventive measures

---

##  Dataset Summary

| Detail             | Value              |
|--------------------|--------------------|
| Rows               | 6,362,620          |
| Columns            | 11                 |
| Fraud cases        | 8,213 (0.13%)      |
| Non-fraud cases    | 6,354,407 (99.87%) |
| Missing values     | 6 rows (dropped)   |

### Features:
- `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- Dropped: `nameOrig`, `nameDest` (IDs)

---

##  Tech Stack

- Python (Pandas, NumPy)
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- Jupyter Notebook (GitHub Codespaces)

---

##  Project Workflow

### 🔹 STEP 1: Data Cleaning
- Removed 6 rows with missing values
- Converted target columns (`isFraud`, `isFlaggedFraud`) to `int`

### 🔹 STEP 2: Exploratory Data Analysis (EDA)
- Top transaction types: `CASH_OUT`, `TRANSFER`
- No fraud in `PAYMENT`, `DEBIT`, `CASH_IN`
- 98.05% of frauds had `newbalanceOrig = 0`
- Identified 338,078 high-value outliers in `amount` (5.3%)

### 🔹 STEP 3: Feature Engineering
- Dropped ID columns (`nameOrig`, `nameDest`)
- One-hot encoded `type` column
- Final features:  
  `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`,  
  `oldbalanceDest`, `newbalanceDest`, `type_CASH_OUT`, `type_TRANSFER`, `isFlaggedFraud`

---

##  STEP 4–8: Model Training & Evaluation

###  STEP 4: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

| Set      | Size   |
| -------- | ------ |
| Training | 70,000 |
| Testing  | 30,000 |

STEP 5: Class Imbalance Handling

from imblearn.over_sampling import SMOTE
X_train_bal, y_train_bal = SMOTE().fit_resample(X_train, y_train)
Original frauds: 0.13%

After SMOTE: 50/50 balance (e.g., 99 fraud, 99 non-fraud)

STEP 6: Model Building

model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)
Chose Random Forest for its robustness & explainability

No scaling needed

STEP 7: Model Evaluation

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 Confusion Matrix:

[[29958     0]
 [    9    33]]
 Final Metrics:
Metric	Score
Accuracy	0.9997
Precision	1.0000
Recall	0.7857
F1-Score	0.8800

 Interpretation:
High precision (1.0) = No false positives
Good recall (0.79) = Most frauds caught
Balanced F1-score (0.88) = Production-ready performance

 STEP 8: Key Insights & Logic
 Top Fraud Indicators:
Feature	Insight
Only TRANSFER and CASH_OUT used in fraud
amount	Fraud transactions often have high values
newbalanceOrig	98% of frauds left sender's balance = 0
oldbalanceOrg	Fraud starts with non-zero sender balance

 Feature Importance (Random Forest):
Feature	Importance
oldbalanceOrg	29.4%
newbalanceOrig	19.5%
amount	16.6%
type_TRANSFER	8.6%
step	7.1%

 STEP 9: Preventive Measures
 Block instant transfers > ₹50,000
 Flag accounts with newbalanceOrig = 0
 Watch for transfers to accounts with newbalanceDest = 0
 Rate-limit users sending multiple transactions in minutes
 Assign fraud risk scores to destination accounts

TEP 10: Monitoring Strategy
Metric	How Often	Why
Recall	Weekly	Catch model degradation
Positives	Weekly	Avoid customer frustration
Financial Impact	Monthly	Ensure cost-effectiveness
Model Retraining	Quarterly	Handle fraud pattern shifts

Use dashboards to track recall and alert thresholds.

Resume Bullet
vbnet
Copy
Edit
Built a fraud detection model using Random Forest with 99.97% accuracy and 0.88 F1-score; handled 6.3M+ transactions, applied SMOTE to balance rare frauds, and identified key risk patterns for prevention.
How to Run
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook main.ipynb
File Structure
bash
Copy
Edit
├── main.ipynb            # Complete code and analysis
├── requirements.txt      # Dependencies
├── README.md             # Project summary (this file)
├── Data Dictionary.txt   # Column details

Author
Saloni
Data Science & Machine Learning Intern
______________

