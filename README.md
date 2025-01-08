**Credit Risk Analysis**

**Description :**
This project involves Credit Risk Analysis to predict loan default risks using financial data. It aims to assist financial institutions in making data-driven decisions, 
improving loan approval strategies, and reducing default rates.
The notebook walks through the entire pipeline from data preprocessing to model evaluation, offering valuable insights into credit risk management.

**Dataset Overview :**
The dataset used for this Credit Risk Analysis consists of 80,000 rows and a variety of features related to borrower financial data. 
Features like LoanAmount, Income, DebtToIncomeRatio, CreditScore, etc.

**Target Variable :**
Defaulted:
The target variable is a binary classification that indicates whether a borrower has defaulted on the loan or not.
0 = Not Defaulted
1 = Defaulted

**Imbalance in Data :**
The dataset is imbalanced, with significantly fewer defaults (1) than non-defaults (0). This imbalance can cause models to perform poorly on the minority class (defaults). 
To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the classes and ensure the model performs well in predicting defaults.

**Features**
Exploratory Data Analysis (EDA):
Identifies trends, anomalies, and data distributions to guide feature engineering.

**Feature Engineering :**
Created new features such as CreditScoreCategory and LoanIncomeRatio.
Handled outliers using IQR-based capping for robust modeling.

**Model Building and Evaluation:**
Includes the following models :
Logistic Regression
Random Forest Classifier
XGBoost
Gradient Boosting Classifier
Evaluated using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

**Important Findings :**
Gradient Boosting Classifier (non-resampled) showed the best performance with:
Accuracy: 0.8972
F1 Score: 0.8355
High precision and recall, indicating the model's ability to identify defaults effectively.
XGBoost (both resampled and non-resampled) had a comparable ROC-AUC (0.9547), but slightly worse accuracy and F1 scores than Gradient Boosting.
Random Forest performed well but had marginally worse recall and F1 scores compared to Gradient Boosting and XGBoost.
Logistic Regression performed poorly, especially with resampled data, across all evaluation metrics.

**Best Model Selection :**
The Gradient Boosting Classifier (non-resampled) was selected as the best model due to its ideal balance of ROC-AUC, F1 score, recall, and accuracy.
This model has a strong recall, ensuring that a large percentage of defaulters are captured.
The high ROC-AUC indicates strong performance in distinguishing between the default and non-default classes.
Random Forest may be used as an alternative if interpretability is prioritized, but for best performance, Gradient Boosting is recommended.

**Interactive Power BI Dashboards :**
Insights from the analysis are presented through dashboards for better decision-making.
