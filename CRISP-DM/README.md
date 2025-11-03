# CRISP-DM: Customer Churn Prediction in Telecom Industry

## 📋 Project Overview

This project implements the complete **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to predict customer churn in the telecommunications industry.

### Dataset
- **Source**: Kaggle - Telco Customer Churn Dataset
- **Size**: 7,043 customers
- **Features**: 21 attributes including demographics, services, and account information
- **Target**: Churn (Yes/No)

## 🔄 CRISP-DM Phases

### Phase 1: Business Understanding
**Objective**: Predict which customers are likely to churn to enable proactive retention strategies.

**Business Questions**:
- What factors contribute most to customer churn?
- Can we predict churn with sufficient accuracy to justify intervention costs?
- What is the expected ROI of a churn prevention program?

**Success Criteria**:
- Model accuracy > 80%
- Precision > 75% (minimize false positives)
- Recall > 70% (catch most churners)
- Actionable insights for retention strategies

### Phase 2: Data Understanding
**Activities**:
- Initial data collection and exploration
- Data quality assessment
- Exploratory Data Analysis (EDA)
- Hypothesis generation

**Key Findings**:
- 26.5% churn rate (class imbalance)
- Month-to-month contracts show higher churn
- Fiber optic customers have higher churn rates
- Senior citizens more likely to churn

### Phase 3: Data Preparation
**Data Cleaning**:
- Handle missing values (TotalCharges)
- Convert data types
- Remove duplicates

**Feature Engineering**:
- Tenure groups (new, medium, long-term)
- Average monthly charges
- Service usage patterns
- Contract value indicators

**Data Transformation**:
- One-hot encoding for categorical variables
- Standardization for numerical features
- Train-test split (80-20)
- SMOTE for handling class imbalance

### Phase 4: Modeling
**Models Implemented**:
1. Logistic Regression (Baseline)
2. Random Forest Classifier
3. Gradient Boosting (XGBoost)
4. LightGBM
5. Neural Network (Deep Learning)

**Model Selection Criteria**:
- Cross-validation performance
- Business metric alignment
- Interpretability
- Computational efficiency

### Phase 5: Evaluation
**Evaluation Metrics**:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Feature Importance Analysis
- SHAP Values for interpretability

**Business Impact Analysis**:
- Cost-benefit analysis
- Expected savings from churn prevention
- Optimal intervention threshold

### Phase 6: Deployment
**Deployment Strategy**:
- REST API using FastAPI
- Batch prediction pipeline
- Real-time scoring endpoint
- Model monitoring dashboard
- A/B testing framework

**Maintenance Plan**:
- Monthly model retraining
- Performance monitoring
- Data drift detection
- Feedback loop integration

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.2% | 67.3% | 55.8% | 61.0% | 0.847 |
| Random Forest | 79.8% | 65.1% | 48.9% | 55.9% | 0.835 |
| XGBoost | 82.1% | 69.8% | 60.2% | 64.6% | 0.868 |
| LightGBM | **83.4%** | **71.2%** | **62.5%** | **66.6%** | **0.879** |
| Neural Network | 81.5% | 68.5% | 58.7% | 63.2% | 0.861 |

**Best Model**: LightGBM

## 🎯 Key Insights

1. **Contract Type**: Month-to-month contracts have 3x higher churn rate
2. **Tenure**: Customers with < 6 months tenure are high-risk
3. **Internet Service**: Fiber optic users churn more (likely due to pricing)
4. **Payment Method**: Electronic check users show higher churn
5. **Support Services**: Lack of tech support correlates with churn

## 💼 Business Recommendations

1. **Retention Program**: Target month-to-month contract customers with incentives
2. **Onboarding**: Enhanced support for first 6 months
3. **Pricing Strategy**: Review fiber optic pricing competitiveness
4. **Payment Options**: Encourage automatic payment methods
5. **Value-Added Services**: Bundle tech support with premium services

## 🚀 Deployment Instructions

### Local Deployment
```bash
cd CRISP-DM
python deployment/api.py
```

### Docker Deployment
```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

### API Usage
```python
import requests

data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 📁 Project Structure

```
CRISP-DM/
├── README.md
├── notebook.ipynb                 # Main Colab notebook
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   └── data_dictionary.md
├── models/
│   ├── lightgbm_model.pkl
│   ├── preprocessor.pkl
│   └── model_card.md
├── reports/
│   ├── eda_report.html
│   ├── model_evaluation.pdf
│   └── business_impact.pdf
├── deployment/
│   ├── api.py
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
└── medium_article.md
```

## 🔗 Resources

- [CRISP-DM Guide](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview)
- [Dataset Source](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Medium Article](./medium_article.md)

## 📈 Future Enhancements

1. Implement customer lifetime value (CLV) prediction
2. Add real-time streaming predictions
3. Develop personalized retention strategies
4. Integrate with CRM systems
5. Build automated alert system for high-risk customers

## 👨‍💻 AI-Assisted Development

This project was developed with iterative AI critique using:
- **Critic Persona**: "World-renowned data science expert and CRISP-DM authority with multiple award-winning books"
- **Revisions**: Multiple iterations per phase ensuring depth and completeness
- **Validation**: Manual verification of all generated code and analyses

---

**Author**: Nitish  
**Date**: November 2024  
**Course**: Advanced Data Mining - Assignment 4
