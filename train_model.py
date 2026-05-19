import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score, f1_score
from imblearn.combine import SMOTETomek
import joblib  
import os
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ==========================================

def load_data(file_path):
    """Load the Telco customer churn dataset"""
    df = pd.read_csv(file_path)
    return df.copy()

def clean_data(df):

    """Clean and preprocess the dataset"""
    pro_df = df.copy()
    
    before = pro_df.shape[0]
    pro_df = pro_df.drop_duplicates().copy()
    after = pro_df.shape[0]
    

    pro_df['TotalCharges'] = pro_df['TotalCharges'].astype(str)
    pro_df['TotalCharges'] = pro_df['TotalCharges'].str.strip()
    pro_df['TotalCharges'] = pro_df['TotalCharges'].str.replace(r'[^0-9.]', '', regex=True)
    pro_df['TotalCharges'] = pd.to_numeric(pro_df['TotalCharges'], errors='coerce')
    

    num_cols = pro_df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = pro_df.select_dtypes(include=['object']).columns
    
    for col in num_cols:
        if pro_df[col].isnull().sum() > 0:
            median_value = pro_df[col].median()
            pro_df[col] = pro_df[col].fillna(median_value)

    for col in cat_cols:
        if pro_df[col].isnull().sum() > 0:
            pro_df[col] = pro_df[col].fillna("Unknown")
    

    if pro_df['SeniorCitizen'].dtype != 'object':
        pro_df['SeniorCitizen'] = pro_df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    

    for col in pro_df.select_dtypes(include='object').columns:
        pro_df[col] = pro_df[col].str.strip()
    
    return pro_df

def create_features(df):
    """Create new features for analysis"""
    pro_df = df.copy()
    

    pro_df['AvgMonthlySpend'] = pro_df['TotalCharges'] / (pro_df['tenure'] + 1)
    
    
    services = [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    pro_df['ServicesCount'] = (pro_df[services] == 'Yes').sum(axis=1)
    
    
    pro_df['TenureGroup'] = pd.cut(
        pro_df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['New', 'Medium', 'Long', 'Very Long']
    )
    
    
    threshold = pro_df['MonthlyCharges'].median()
    pro_df['HighValue'] = (pro_df['MonthlyCharges'] > threshold).astype(int)
    
    
    pro_df['IsLongContract'] = pro_df['Contract'].isin(['One year', 'Two year']).astype(int)
    

    pro_df['RiskScore'] = (
        (pro_df['Contract'] == 'Month-to-month').astype(int) +
        (pro_df['MonthlyCharges'] > pro_df['MonthlyCharges'].median()).astype(int) +
        (pro_df['tenure'] < 12).astype(int)
    )
    
    return pro_df

def get_dataset_info(df):
    """Return dataset information as a dictionary"""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return info

# ==========================================
# FEATURE SELECTION FUNCTIONS
# ==========================================

def correlation_analysis(df):
    """Perform correlation analysis with churn"""
    df_fs = df.copy()
    df_fs['Churn'] = df_fs['Churn'].map({'Yes': 1, 'No': 0})
    
    num_df = df_fs.select_dtypes(include=['int64', 'float64'])
    corr = num_df.corr()['Churn'].sort_values(ascending=False)
    return corr

def chi_square_analysis(df):
    """Perform chi-square feature selection"""
    df_fs = df.copy()
    df_fs['Churn'] = df_fs['Churn'].map({'Yes': 1, 'No': 0})
    
    chi_df = df_fs.copy()
    
    for col in chi_df.columns:
        if chi_df[col].dtype == 'object' or str(chi_df[col].dtype) == 'category':
            chi_df[col] = chi_df[col].astype(str)
    
    cat_cols = chi_df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in cat_cols:
        chi_df[col] = le.fit_transform(chi_df[col])
    
    X_chi = chi_df.drop(columns=['Churn', 'customerID'], errors='ignore')
    y_chi = chi_df['Churn']
    X_chi = X_chi.astype(float)
    X_chi = X_chi.clip(lower=0)
    
    chi_scores, _ = chi2(X_chi, y_chi)
    chi_result = pd.Series(chi_scores, index=X_chi.columns)
    chi_result = chi_result.sort_values(ascending=False)
    
    return chi_result

def random_forest_importance(df):
    """Get feature importance using Random Forest"""
    df_fs = df.copy()
    df_fs['Churn'] = df_fs['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df_fs.drop(columns=['Churn', 'customerID'], errors='ignore')
    y = df_fs['Churn']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_encoded, y)
    
    rf_importance = pd.Series(rf.feature_importances_, index=X_encoded.columns)
    return rf_importance.sort_values(ascending=False)

def get_pivot_tables(df):
    """Generate pivot tables for analysis"""
    pivot1 = pd.pivot_table(
        df,
        values='MonthlyCharges',
        index='Churn',
        aggfunc='mean'
    )
    
    pivot2 = pd.pivot_table(
        df,
        values='customerID',
        index='Contract',
        columns='Churn',
        aggfunc='count'
    )
    
    return pivot1, pivot2

# ==========================================
# MODEL TRAINING FUNCTIONS
# ==========================================

def prepare_data_for_model(df):
    """Prepare data for model training"""
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
    y = df['Churn']
    
    # One-Hot Encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train baseline model without handling imbalance"""
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Get metrics
    baseline_recall = recall_score(y_test, y_pred_baseline, pos_label=1)
    baseline_precision = precision_score(y_test, y_pred_baseline, pos_label=1)
    baseline_f1 = f1_score(y_test, y_pred_baseline, pos_label=1)
    
    return {
        'model': baseline_model,
        'predictions': y_pred_baseline,
        'recall': baseline_recall,
        'precision': baseline_precision,
        'f1': baseline_f1,
        'report': classification_report(y_test, y_pred_baseline, target_names=['No Churn', 'Churn'])
    }

def apply_smote_tomek(X_train, y_train):
    """Apply SMOTETomek balancing technique"""
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def train_optimized_model(X_train_res, y_train_res):
    """Train optimized Random Forest model"""
    optimized_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    optimized_model.fit(X_train_res, y_train_res)
    return optimized_model

def find_optimal_threshold(model, X_test, y_test):
    """Find optimal decision threshold"""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.25, 0.51, 0.05)
    results = []
    
    for thresh in thresholds:
        y_pred_tuned = (y_proba >= thresh).astype(int)
        rec = recall_score(y_test, y_pred_tuned, pos_label=1)
        prec = precision_score(y_test, y_pred_tuned, pos_label=1)
        f1 = f1_score(y_test, y_pred_tuned, pos_label=1)
        
        results.append({
            'threshold': thresh,
            'recall': rec,
            'precision': prec,
            'f1': f1
        })
    
    # Select best threshold (prioritize Recall while keeping Precision >= 0.50)
    best_result = None
    best_score = 0
    
    for res in results:
        if res['precision'] >= 0.50:
            if res['recall'] > best_score:
                best_score = res['recall']
                best_result = res
    
    if best_result is None:
        best_result = max(results, key=lambda x: x['f1'])
    
    return best_result['threshold'], y_proba, results

def run_complete_pipeline(file_path):
    """Run the complete data preprocessing and model training pipeline"""
    # Load and process data
    df = load_data(file_path)
    df_clean = clean_data(df)
    df_features = create_features(df_clean)
    
    # Get feature importance
    corr_results = correlation_analysis(df_features)
    chi_results = chi_square_analysis(df_features)
    rf_results = random_forest_importance(df_features)
    
    # Prepare data for model
    X, y = prepare_data_for_model(df_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train baseline model
    baseline_results = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # Apply SMOTETomek
    X_train_res, y_train_res = apply_smote_tomek(X_train, y_train)
    
    # Train optimized model
    optimized_model = train_optimized_model(X_train_res, y_train_res)
    
    # Find optimal threshold
    best_threshold, y_proba, threshold_results = find_optimal_threshold(optimized_model, X_test, y_test)
    
    # Final predictions
    y_pred_final = (y_proba >= best_threshold).astype(int)
    
    # Final metrics
    final_recall = recall_score(y_test, y_pred_final, pos_label=1)
    final_precision = precision_score(y_test, y_pred_final, pos_label=1)
    final_f1 = f1_score(y_test, y_pred_final, pos_label=1)
    
    return {
        'baseline': baseline_results,
        'optimized_model': optimized_model,
        'best_threshold': best_threshold,
        'final_predictions': y_pred_final,
        'final_metrics': {
            'recall': final_recall,
            'precision': final_precision,
            'f1': final_f1
        },
        'feature_importance': {
            'correlation': corr_results,
            'chi_square': chi_results,
            'random_forest': rf_results
        },
        'threshold_results': threshold_results,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_res': X_train_res,
        'y_train_res': y_train_res,
        'feature_columns': X.columns.tolist()  # Save feature columns
    }

def predict_churn(model, threshold, new_data, feature_columns):
    """Predict churn for new data"""
    # Preprocess new data
    new_data_clean = clean_data(new_data)
    new_data_features = create_features(new_data_clean)
    X_new = pd.get_dummies(new_data_features.drop(columns=['customerID'], errors='ignore'), drop_first=True)
    
    # Align columns with training data
    for col in feature_columns:
        if col not in X_new.columns:
            X_new[col] = 0
    
    X_new = X_new[feature_columns]
    
    # Get predictions
    y_proba = model.predict_proba(X_new)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    return y_pred, y_proba

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def plot_class_distribution(y_train, y_train_res=None):
    """Plot class distribution before and after balancing"""
    fig, axes = plt.subplots(1, 2 if y_train_res is not None else 1, figsize=(14, 5))
    
    if y_train_res is not None:
        # Before plot
        class_counts = y_train.value_counts()
        axes[0].pie(class_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.05))
        axes[0].set_title('Class Distribution - BEFORE Balancing', fontsize=13, fontweight='bold')
        
        # After plot
        balanced_counts = y_train_res.value_counts()
        axes[1].pie(balanced_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.05))
        axes[1].set_title('Class Distribution - AFTER SMOTETomek', fontsize=13, fontweight='bold')
    else:
        class_counts = y_train.value_counts()
        axes.pie(class_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                 colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.05))
        axes.set_title('Class Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_final, threshold):
    """Plot confusion matrices for baseline and final models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_baseline, 
        display_labels=['No Churn', 'Churn'],
        ax=axes[0], cmap='Reds'
    )
    axes[0].set_title('Confusion Matrix - BASELINE', fontsize=12, fontweight='bold')
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_final,
        display_labels=['No Churn', 'Churn'],
        ax=axes[1], cmap='Greens'
    )
    axes[1].set_title(f'Confusion Matrix - FINAL (Threshold={threshold:.2f})', 
                      fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(rf_importance, top_n=15):
    """Plot top N feature importances"""
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = rf_importance.head(top_n)
    
    ax.barh(range(len(top_features)), top_features.values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)', fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# ==========================================
# SAVE MODEL FUNCTION
# ==========================================

def save_model(model, feature_columns, model_filename="model.pkl", columns_filename="columns.pkl"):
    """Save the trained model and feature columns"""
    try:
        joblib.dump(model, model_filename)
        joblib.dump(feature_columns, columns_filename)
        
        print("=" * 60)
        print("✅ MODEL SAVED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Model saved as: {model_filename}")
        print(f"📁 Feature columns saved as: {columns_filename}")
        print(f"📍 Current working directory: {os.getcwd()}")
        print("=" * 60)
        
        return True
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return False

def load_saved_model(model_filename="model.pkl", columns_filename="columns.pkl"):
    """Load saved model and feature columns"""
    try:
        model = joblib.load(model_filename)
        feature_columns = joblib.load(columns_filename)
        
        print("=" * 60)
        print("✅ MODEL LOADED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Model loaded from: {model_filename}")
        print(f"📁 Feature columns loaded from: {columns_filename}")
        print("=" * 60)
        
        return model, feature_columns
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

# ==========================================
# EXAMPLE USAGE (for GUI integration)
# ==========================================

if __name__ == "__main__":
    # Example of how to use these functions in a GUI
    file_path = r"Telco-Customer-Churn.csv"
    
    # Run complete pipeline
    print("\n🚀 Starting pipeline execution...\n")
    results = run_complete_pipeline(file_path)
    
    print("\n=== PIPELINE RESULTS ===\n")
    print(f"📊 Baseline Recall: {results['baseline']['recall']:.3f}")
    print(f"📊 Final Recall: {results['final_metrics']['recall']:.3f}")
    print(f"🎯 Optimal Threshold: {results['best_threshold']:.2f}")
    print(f"📈 Final F1-Score: {results['final_metrics']['f1']:.3f}")
    
    print("\n" + "=" * 60)
    print("🎯 Classification Report (Final Model):")
    print("=" * 60)
    print(classification_report(results['y_test'], results['final_predictions'], 
                                target_names=['No Churn', 'Churn']))
    
    # Save the model and feature columns
    print("\n💾 Saving model...\n")
    save_model(
        results['optimized_model'], 
        results['feature_columns'],
        model_filename="model.pkl",
        columns_filename="columns.pkl"
    )
    
    # Optional: Test loading the saved model
    print("\n🔄 Testing model loading...\n")
    loaded_model, loaded_columns = load_saved_model("model.pkl", "columns.pkl")
    
    if loaded_model is not None:
        print("\n✅ Model verification successful! Model is ready for GUI integration.")
        
        # Example prediction (demonstration)
        print("\n📊 Example prediction with loaded model:")
        print("-" * 40)
        
        # Get a sample from test set
        sample_data = results['X_test'].iloc[[0]]
        sample_proba = loaded_model.predict_proba(sample_data)[:, 1]
        sample_pred = (sample_proba >= results['best_threshold']).astype(int)
        
        print(f"Sample prediction: {'Churn' if sample_pred[0] == 1 else 'No Churn'}")
        print(f"Churn probability: {sample_proba[0]:.3f}")
        print(f"Decision threshold: {results['best_threshold']:.2f}")
        print("-" * 40)
    
    # Example of making predictions on new data
    # new_customer_data = pd.read_csv("new_customers.csv")
    # predictions, probabilities = predict_churn(
    #     results['optimized_model'], 
    #     results['best_threshold'], 
    #     new_customer_data,
    #     results['feature_columns']
    # )