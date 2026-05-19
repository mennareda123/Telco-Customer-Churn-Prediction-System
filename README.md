# Telco Customer Churn Prediction System

## 📋 Project Overview

This project is an end-to-end machine learning solution for predicting customer churn in the telecommunications industry. The system analyzes customer behavior, account details, and service subscriptions to identify customers at risk of leaving, enabling proactive retention strategies.

## 📊 Dataset

The dataset contains **7,043 customer records** with 21 features including:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Account Info**: tenure, Contract type, PaperlessBilling, PaymentMethod
- **Services**: PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
- **Charges**: MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)

## 🏗️ Repository Structure

```
telco-churn-prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv         
├── models/
│   ├── model.pkl                       
│   └── columns.pkl                     
├── notebooks/
│   └── Final_project.ipynb            
├── src/
│   ├── train_model.py                   
│   └── churn_gui.py                      # Streamlit GUI application
├── docs/
│   └── Report.pdf                       
├── requirements.txt                                              
└──README.md                           
```


## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/mennareda123/telco-churn-prediction.git
cd telco-churn-prediction
```

2. **Run model training:**
```bash
python src/train_model.py
```

3. **Launch the GUI application:**
```bash
streamlit run src/churn_gui.py
```

## 🔧 Data Processing Pipeline

### Data Cleaning
- Removed duplicate records
- Handled missing values (median for numerical, 'Unknown' for categorical)
- Corrected data types (TotalCharges converted to numeric)
- Standardized text fields (stripped whitespace)

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `AvgMonthlySpend` | TotalCharges / (tenure + 1) |
| `ServicesCount` | Number of services customer subscribes to |
| `TenureGroup` | Categorical: New, Medium, Long, Very Long |
| `HighValue` | MonthlyCharges > median |
| `IsLongContract` | Contract is One year or Two year |
| `RiskScore` | Sum of churn risk indicators (0-3) |

##  Model Architecture

1. **Baseline Model:** Random Forest (imbalanced data)
2. **Imbalance Handling:** SMOTETomek (oversampling + cleaning)
3. **Optimized Model:** Random Forest with tuned hyperparameters
4. **Threshold Optimization:** Decision threshold adjusted to 0.35 (prioritizing recall)

### Model Performance

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| **Recall (Churn)** | 49.7% | **79.1%** | +59.1% |
| **Precision (Churn)** | 63.7% | 50.9% | -20.0% |
| **F1-Score (Churn)** | 0.559 | **0.620** | +11.0% |
| **Accuracy** | 79% | 74% | -5% |

##  Key Insights

1. **Contract type** is the strongest predictor of churn
   - Month-to-month customers have highest churn rate
   
2. **Tenure** is inversely correlated with churn
   - Customers with <12 months tenure are at highest risk
   
3. **High monthly charges** increase churn probability
   
4. **Fiber optic** customers churn more than DSL customers

##  GUI Features

The Streamlit application includes:

- **Home Page**: Dashboard with key metrics and business impact
- **Single Prediction**: Individual customer risk assessment
- **Batch Prediction**: Upload CSV for bulk predictions
- **Analytics Dashboard**: Interactive visualizations
- **Model Info**: Performance metrics and feature importance

##  Business Recommendations

1.  **Target month-to-month contract customers** with retention offers
2.  **Proactively engage customers** with high monthly charges
3.  **Focus retention efforts** on customers with less than 12 months tenure
4.  **Offer bundled services** to increase switching costs
5.  **Implement early warning system** using this model

##  Technologies Used

- **Python 3.10+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **Imbalanced-learn** - SMOTETomek for class balancing
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application GUI
- **Joblib** - Model serialization

##  License

This project is for educational purposes.

##  Author

Developed as part of a data science project for customer churn prediction.

---

** Note:** Make sure the CSV file (`Telco-Customer-Churn.csv`) is in the same directory when running the training script, or update the file path accordingly.
