# Data Dictionary: Telco Customer Churn Dataset

## Dataset Overview
- **Source**: IBM Sample Data / Kaggle
- **Records**: 7,043 customers
- **Features**: 21 columns
- **Target**: Churn (Yes/No)
- **Time Period**: Not specified (cross-sectional data)

---

## Feature Descriptions

### Customer Demographics

| Feature | Type | Description | Values | Notes |
|---------|------|-------------|--------|-------|
| **customerID** | String | Unique customer identifier | e.g., "7590-VHVEG" | Primary key, not used in modeling |
| **gender** | Categorical | Customer gender | Male, Female | Binary categorical |
| **SeniorCitizen** | Binary | Whether customer is senior citizen | 0 = No, 1 = Yes | Encoded as integer |
| **Partner** | Categorical | Whether customer has a partner | Yes, No | Marital status indicator |
| **Dependents** | Categorical | Whether customer has dependents | Yes, No | Family status indicator |

### Account Information

| Feature | Type | Description | Values | Notes |
|---------|------|-------------|--------|-------|
| **tenure** | Numeric | Months customer has been with company | 0-72 | Key feature for churn prediction |
| **Contract** | Categorical | Type of contract | Month-to-month, One year, Two year | Strong predictor of churn |
| **PaperlessBilling** | Categorical | Whether customer uses paperless billing | Yes, No | Digital engagement indicator |
| **PaymentMethod** | Categorical | Payment method used | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | 4 categories |
| **MonthlyCharges** | Numeric | Monthly bill amount | $18.25 - $118.75 | Continuous variable |
| **TotalCharges** | Numeric | Total amount charged over tenure | $18.80 - $8,684.80 | May have missing values for new customers |

### Services Subscribed

| Feature | Type | Description | Values | Notes |
|---------|------|-------------|--------|-------|
| **PhoneService** | Categorical | Whether customer has phone service | Yes, No | Basic service |
| **MultipleLines** | Categorical | Whether customer has multiple phone lines | Yes, No, No phone service | Conditional on PhoneService |
| **InternetService** | Categorical | Type of internet service | DSL, Fiber optic, No | Key service differentiator |
| **OnlineSecurity** | Categorical | Whether customer has online security add-on | Yes, No, No internet service | Conditional on InternetService |
| **OnlineBackup** | Categorical | Whether customer has online backup add-on | Yes, No, No internet service | Conditional on InternetService |
| **DeviceProtection** | Categorical | Whether customer has device protection add-on | Yes, No, No internet service | Conditional on InternetService |
| **TechSupport** | Categorical | Whether customer has tech support add-on | Yes, No, No internet service | Conditional on InternetService |
| **StreamingTV** | Categorical | Whether customer has streaming TV add-on | Yes, No, No internet service | Conditional on InternetService |
| **StreamingMovies** | Categorical | Whether customer has streaming movies add-on | Yes, No, No internet service | Conditional on InternetService |

### Target Variable

| Feature | Type | Description | Values | Notes |
|---------|------|-------------|--------|-------|
| **Churn** | Binary | Whether customer churned | Yes, No | Target variable for prediction |

---

## Feature Statistics

### Numerical Features

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| tenure | 32.4 | 24.6 | 0 | 9 | 29 | 55 | 72 |
| MonthlyCharges | 64.76 | 30.09 | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 |
| TotalCharges | 2283.30 | 2266.77 | 18.80 | 401.45 | 1397.47 | 3794.74 | 8684.80 |

### Categorical Features Distribution

| Feature | Most Common Value | Frequency | Churn Rate |
|---------|-------------------|-----------|------------|
| gender | Male | 50.5% | 26.9% |
| SeniorCitizen | No (0) | 83.9% | 23.6% |
| Partner | No | 51.8% | 33.0% |
| Dependents | No | 70.4% | 31.3% |
| PhoneService | Yes | 90.3% | 26.7% |
| InternetService | Fiber optic | 44.0% | 41.9% |
| Contract | Month-to-month | 55.0% | 42.7% |
| PaperlessBilling | Yes | 59.2% | 33.6% |

---

## Data Quality Issues

### Missing Values
- **TotalCharges**: 11 missing values (0.16%)
  - Reason: New customers with tenure = 0
  - Solution: Impute with 0 or MonthlyCharges

### Inconsistencies
- **TotalCharges**: Originally stored as object/string
  - Contains whitespace for missing values
  - Requires conversion to numeric

### Outliers
- **tenure**: Some customers with 0 months (new signups)
- **MonthlyCharges**: Wide range ($18-$119) but no true outliers
- **TotalCharges**: Highly correlated with tenure (expected)

---

## Feature Engineering Opportunities

### Created Features

1. **tenure_group**
   - Categorical grouping of tenure
   - Bins: 0-12, 12-24, 24-48, 48+ months
   - Captures non-linear relationship with churn

2. **avg_monthly_charges**
   - TotalCharges / tenure
   - Identifies customers with changing pricing
   - Formula: `TotalCharges / (tenure + 1)`

3. **service_count**
   - Count of additional services subscribed
   - Range: 0-8 services
   - Indicator of customer engagement

4. **high_value_customer**
   - Binary flag for high-value customers
   - Criteria: Long tenure + High monthly charges
   - Formula: `(tenure > 24) & (MonthlyCharges > median)`

5. **contract_value**
   - Estimated contract value
   - Formula: `MonthlyCharges * contract_length_months`
   - Helps prioritize retention efforts

### Interaction Features

1. **internet_tech_support**
   - Interaction: InternetService × TechSupport
   - Captures: Fiber users without tech support (high churn)

2. **senior_partner**
   - Interaction: SeniorCitizen × Partner
   - Captures: Senior citizens living alone (higher churn)

---

## Target Variable Analysis

### Churn Distribution
- **No Churn**: 5,174 customers (73.5%)
- **Churn**: 1,869 customers (26.5%)
- **Imbalance Ratio**: 2.77:1

### Churn by Key Features

| Feature | No Churn Rate | Churn Rate |
|---------|---------------|------------|
| **Contract Type** | | |
| Month-to-month | 57.3% | 42.7% |
| One year | 88.6% | 11.4% |
| Two year | 97.1% | 2.9% |
| **Internet Service** | | |
| DSL | 81.0% | 19.0% |
| Fiber optic | 58.1% | 41.9% |
| No | 92.6% | 7.4% |
| **Tenure** | | |
| 0-12 months | 50.2% | 49.8% |
| 13-24 months | 65.1% | 34.9% |
| 25-48 months | 81.7% | 18.3% |
| 49+ months | 93.2% | 6.8% |

---

## Business Insights from Data

### High-Risk Customer Profile
- Month-to-month contract
- Tenure < 12 months
- Fiber optic internet
- No tech support
- Electronic check payment
- No partner or dependents

### Low-Risk Customer Profile
- Two-year contract
- Tenure > 24 months
- Multiple services subscribed
- Automatic payment method
- Has tech support
- Has partner and/or dependents

### Key Churn Drivers (Correlation with Churn)
1. **Contract Type** (r = 0.40): Month-to-month highest risk
2. **Tenure** (r = -0.35): Longer tenure = lower churn
3. **Internet Service** (r = 0.31): Fiber optic higher churn
4. **Tech Support** (r = -0.29): No tech support = higher churn
5. **Payment Method** (r = 0.28): Electronic check higher churn

---

## Data Usage Guidelines

### For Modeling
1. **Drop**: customerID (identifier, not predictive)
2. **Encode**: All categorical variables
3. **Scale**: Numerical features (tenure, charges)
4. **Handle Imbalance**: Use SMOTE or class weights
5. **Split**: 80-20 train-test with stratification

### For Business Analysis
1. **Segment**: By contract type and tenure
2. **Prioritize**: High-value customers at risk
3. **Target**: Month-to-month customers for upgrades
4. **Intervene**: New customers (< 6 months) proactively

### For Deployment
1. **Required Fields**: All features except customerID
2. **Data Types**: Ensure correct types before prediction
3. **Validation**: Check for valid category values
4. **Missing Values**: Handle TotalCharges appropriately

---

## References

- **Dataset Source**: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Kaggle**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Documentation**: [IBM SPSS Modeler](https://www.ibm.com/docs/en/spss-modeler/)

---

*Last Updated: November 2024*
