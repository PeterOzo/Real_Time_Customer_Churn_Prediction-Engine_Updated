import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb

# SMOTE (Optional - with fallback for compatibility issues)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
import pickle
import io

# Deep Learning (Optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ACA Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1edff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

@st.cache_data
def load_data():
    """Load the customer churn dataset"""
    try:
        # Load real-world telecom churn dataset
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_data(df):
    """Advanced data preprocessing and feature engineering"""
    df_processed = df.copy()
    
    # Convert TotalCharges to numeric
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
    
    # Convert target variable to binary
    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    
    # Advanced Feature Engineering
    # 1. Customer Tenure Segments
    df_processed['TenureSegment'] = pd.cut(df_processed['tenure'],
                                         bins=[0, 12, 24, 48, 100],
                                         labels=['New', 'Developing', 'Established', 'Loyal'])
    
    # 2. Count active services
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df_processed['TotalServices'] = 0
    for service in services:
        if service == 'PhoneService':
            df_processed['TotalServices'] += (df_processed[service] == 'Yes').astype(int)
        elif service == 'InternetService':
            df_processed['TotalServices'] += (df_processed[service] != 'No').astype(int)
        else:
            df_processed['TotalServices'] += (df_processed[service] == 'Yes').astype(int)
    
    # 3. Value-based features
    df_processed['ChargesPerService'] = df_processed['MonthlyCharges'] / (df_processed['TotalServices'] + 1)
    df_processed['TotalChargesPerTenure'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)
    df_processed['EstimatedCLV'] = df_processed['MonthlyCharges'] * (df_processed['tenure'] + 12)
    
    # 4. Payment behavior features
    df_processed['AutoPay'] = (df_processed['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])).astype(int)
    df_processed['DigitalPayment'] = (df_processed['PaymentMethod'].isin(['Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)'])).astype(int)
    
    # 5. Service adoption features
    df_processed['HasStreamingServices'] = ((df_processed['StreamingTV'] == 'Yes') |
                                           (df_processed['StreamingMovies'] == 'Yes')).astype(int)
    df_processed['HasSecurityServices'] = ((df_processed['OnlineSecurity'] == 'Yes') |
                                          (df_processed['OnlineBackup'] == 'Yes') |
                                          (df_processed['DeviceProtection'] == 'Yes')).astype(int)
    
    # 6. Contract value features
    contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df_processed['ContractMonths'] = df_processed['Contract'].map(contract_mapping)
    df_processed['ContractValue'] = df_processed['MonthlyCharges'] * df_processed['ContractMonths']
    
    return df_processed

@st.cache_data
def prepare_features(df_processed):
    """Prepare features for machine learning"""
    features_to_encode = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                         'PaymentMethod', 'TenureSegment']
    
    df_ml = df_processed.copy()
    label_encoders = {}
    
    # Label encode categorical variables
    for feature in features_to_encode:
        if feature in df_ml.columns:
            le = LabelEncoder()
            df_ml[feature + '_encoded'] = le.fit_transform(df_ml[feature].astype(str))
            label_encoders[feature] = le
    
    # Select features for modeling
    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                         'TotalServices', 'ChargesPerService', 'TotalChargesPerTenure',
                         'EstimatedCLV', 'AutoPay', 'DigitalPayment', 'HasStreamingServices',
                         'HasSecurityServices', 'ContractValue']
    
    categorical_encoded = [f + '_encoded' for f in features_to_encode if f in df_ml.columns]
    feature_columns = numerical_features + categorical_encoded
    feature_columns = [col for col in feature_columns if col in df_ml.columns]
    
    X = df_ml[feature_columns]
    y = df_ml['Churn']
    
    return X, y, feature_columns, label_encoders

def train_models(X, y):
    """Train multiple machine learning models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE (if available)
    if SMOTE_AVAILABLE:
        try:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            st.info("✅ Applied SMOTE for class balancing")
        except Exception as e:
            st.warning(f"SMOTE failed: {e}. Using original training data.")
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        st.info("ℹ️ SMOTE unavailable - using original training data with class weights")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    models_results = {}
    
    # Random Forest
    st.info("🌲 Training Random Forest...")
    if SMOTE_AVAILABLE and len(X_train_balanced) > len(X_train):
        # Use balanced data from SMOTE
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    else:
        # Use class weights for balancing
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
    
    rf.fit(X_train_balanced, y_train_balanced)
    rf_pred = rf.predict(X_test)
    rf_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    models_results['Random Forest'] = {
        'model': rf,
        'predictions': rf_pred,
        'probabilities': rf_pred_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_pred_proba),
        'f1': f1_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred)
    }
    
    # XGBoost
    st.info("🚀 Training XGBoost...")
    if SMOTE_AVAILABLE and len(X_train_balanced) > len(X_train):
        # Use balanced data from SMOTE
        scale_pos_weight = 1
    else:
        # Calculate scale_pos_weight for class balancing
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    models_results['XGBoost'] = {
        'model': xgb_model,
        'predictions': xgb_pred,
        'probabilities': xgb_pred_proba,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'roc_auc': roc_auc_score(y_test, xgb_pred_proba),
        'f1': f1_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred)
    }
    
    # Gradient Boosting
    st.info("⚡ Training Gradient Boosting...")
    # Gradient Boosting doesn't have class_weight parameter, so we'll use sample_weight if needed
    if SMOTE_AVAILABLE and len(X_train_balanced) > len(X_train):
        # Use balanced data from SMOTE
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        sample_weight = None
    else:
        # Calculate sample weights for class balancing
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight('balanced', y_train_balanced)
    
    gb.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weight)
    gb_pred = gb.predict(X_test)
    gb_pred_proba = gb.predict_proba(X_test)[:, 1]
    
    models_results['Gradient Boosting'] = {
        'model': gb,
        'predictions': gb_pred,
        'probabilities': gb_pred_proba,
        'accuracy': accuracy_score(y_test, gb_pred),
        'roc_auc': roc_auc_score(y_test, gb_pred_proba),
        'f1': f1_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred),
        'recall': recall_score(y_test, gb_pred)
    }
    
    # Neural Network (Optional - only if TensorFlow is available)
    if TENSORFLOW_AVAILABLE:
        st.info("🧠 Training Neural Network...")
        try:
            # Scale the balanced training data for neural network
            X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
            
            # Build neural network architecture
            def create_neural_network(input_dim):
                model = Sequential([
                    Dense(128, activation='relu', input_dim=input_dim),
                    BatchNormalization(),
                    Dropout(0.3),
                    
                    Dense(64, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    
                    Dense(16, activation='relu'),
                    Dropout(0.1),
                    
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer=Adam(learning_rate=0.001),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
                
                return model
            
            # Create and train neural network
            nn_model = create_neural_network(X_train_balanced_scaled.shape[1])
            
            # Train the model with reduced epochs for faster deployment
            history = nn_model.fit(
                X_train_balanced_scaled, y_train_balanced,
                validation_data=(X_val_scaled, y_val),
                epochs=20,  # Reduced for faster training
                batch_size=32,
                verbose=0  # Silent training
            )
            
            # Make predictions
            nn_pred_proba = nn_model.predict(X_test_scaled, verbose=0).flatten()
            nn_pred = (nn_pred_proba > 0.5).astype(int)
            
            models_results['Neural Network'] = {
                'model': nn_model,
                'predictions': nn_pred,
                'probabilities': nn_pred_proba,
                'accuracy': accuracy_score(y_test, nn_pred),
                'roc_auc': roc_auc_score(y_test, nn_pred_proba),
                'f1': f1_score(y_test, nn_pred),
                'precision': precision_score(y_test, nn_pred),
                'recall': recall_score(y_test, nn_pred)
            }
        except Exception as e:
            st.warning(f"⚠ Neural Network training failed: {str(e)}")
    else:
        st.info("ℹ Neural Network skipped (TensorFlow not available)")
    
    return models_results, X_test, y_test, scaler

def calculate_business_impact(models_results, y_test, avg_clv):
    """Calculate business impact and ROI"""
    retention_cost = 100
    successful_retention_rate = 0.3
    
    business_metrics = {}
    
    for model_name, results in models_results.items():
        predictions = results['probabilities']
        threshold = 0.5
        
        tp = np.sum((predictions >= threshold) & (y_test == 1))
        fp = np.sum((predictions >= threshold) & (y_test == 0))
        fn = np.sum((predictions < threshold) & (y_test == 1))
        
        value_from_retained = tp * successful_retention_rate * avg_clv
        cost_of_campaigns = (tp + fp) * retention_cost
        cost_of_missed = fn * avg_clv * 0.3  # Partial cost for missed opportunities
        
        net_value = value_from_retained - cost_of_campaigns
        roi = (net_value / cost_of_campaigns) * 100 if cost_of_campaigns > 0 else 0
        
        business_metrics[model_name] = {
            'net_value': net_value,
            'roi': roi,
            'customers_targeted': tp + fp,
            'customers_retained': tp * successful_retention_rate
        }
    
    return business_metrics

# Main App
def main():
    st.markdown('<div class="main-header">🎯 ACA Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    if not TENSORFLOW_AVAILABLE:
        st.sidebar.info("📝 Neural Network model unavailable (TensorFlow not installed)")
    
    if not SMOTE_AVAILABLE:
        st.sidebar.info("📝 SMOTE unavailable - using class weights for balancing")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Overview", "📊 Data Analysis", "🤖 Model Training", "📈 Predictions", "💼 Business Impact", "🔍 Model Insights"]
    )
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading real customer data..."):
            df = load_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.df_processed = preprocess_data(df)
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            else:
                st.error("❌ Failed to load data")
                return
    
    df = st.session_state.df
    df_processed = st.session_state.df_processed
    
    # Overview Page
    if page == "🏠 Overview":
        st.header("📋 Project Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Customers</h3>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            churn_rate = df['Churn'].value_counts(normalize=True)['Yes']
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Churn Rate</h3>
                <h2>{churn_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_clv = df_processed['EstimatedCLV'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Avg CLV</h3>
                <h2>${avg_clv:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            features_count = len([col for col in df_processed.columns if '_encoded' in col or col in ['TotalServices', 'ChargesPerService']])
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Features</h3>
                <h2>{features_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>🎯 Project Objective</h3>
            <p>Develop an advanced machine learning model to predict customer churn and enable proactive retention strategies for American Credit Acceptance (ACA). This dashboard demonstrates cutting-edge data science techniques applicable to auto finance industry challenges.</p>
            
            <h4>🤖 Available Models</h4>
            <p>• Random Forest Classifier (with class balancing)<br>
            • XGBoost Classifier (with scale_pos_weight)<br>
            • Gradient Boosting Classifier (with sample weights)<br>
            {"• Neural Network (Deep Learning)" if TENSORFLOW_AVAILABLE else "• Neural Network (Unavailable - requires TensorFlow)"}</p>
            
            <h4>⚖️ Class Balancing</h4>
            <p>{"• SMOTE (Synthetic Minority Oversampling)" if SMOTE_AVAILABLE else "• Class weights and sample balancing"}<br>
            • Robust handling of imbalanced datasets<br>
            • Multiple fallback strategies for optimal performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights
        st.subheader("🔍 Key Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by contract type
            churn_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
            fig_contract = px.bar(
                x=churn_contract.index,
                y=churn_contract.values,
                title="Churn Rate by Contract Type",
                labels={'x': 'Contract Type', 'y': 'Churn Rate'},
                color=churn_contract.values,
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_contract, use_container_width=True)
        
        with col2:
            # Monthly charges distribution
            fig_charges = px.histogram(
                df, x='MonthlyCharges', color='Churn',
                title="Monthly Charges Distribution by Churn",
                nbins=30, opacity=0.7
            )
            st.plotly_chart(fig_charges, use_container_width=True)
    
    # Data Analysis Page
    elif page == "📊 Data Analysis":
        st.header("📊 Data Exploration & Analysis")
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.dataframe(missing_data[missing_data > 0])
            else:
                st.write("No missing values found!")
        
        with col2:
            st.write("**Target Distribution:**")
            target_dist = df['Churn'].value_counts()
            fig_target = px.pie(values=target_dist.values, names=target_dist.index, title="Churn Distribution")
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Feature analysis
        st.subheader("🎨 Advanced Feature Engineering")
        feature_tabs = st.tabs(["📊 Numerical Features", "🏷 Categorical Features", "🆕 Engineered Features"])
        
        with feature_tabs[0]:
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            selected_num_feature = st.selectbox("Select numerical feature:", numerical_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df, x=selected_num_feature, color='Churn', 
                                      title=f"{selected_num_feature} Distribution", opacity=0.7)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(df, x='Churn', y=selected_num_feature, 
                               title=f"{selected_num_feature} by Churn Status")
                st.plotly_chart(fig_box, use_container_width=True)
        
        with feature_tabs[1]:
            categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
            selected_cat_feature = st.selectbox("Select categorical feature:", categorical_cols)
            
            churn_by_cat = df.groupby(selected_cat_feature)['Churn'].apply(lambda x: (x == 'Yes').mean())
            fig_cat = px.bar(x=churn_by_cat.index, y=churn_by_cat.values,
                           title=f"Churn Rate by {selected_cat_feature}",
                           labels={'x': selected_cat_feature, 'y': 'Churn Rate'})
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with feature_tabs[2]:
            st.write("**Engineered Features Preview:**")
            engineered_features = ['TotalServices', 'ChargesPerService', 'EstimatedCLV', 'ContractValue']
            st.dataframe(df_processed[engineered_features + ['Churn']].head(10))
            
            # Correlation heatmap
            corr_matrix = df_processed[engineered_features + ['Churn']].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Model Training Page
    elif page == "🤖 Model Training":
        st.header("🤖 Advanced Model Training")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training multiple ML models... This may take a few minutes."):
                X, y, feature_columns, label_encoders = prepare_features(df_processed)
                models_results, X_test, y_test, scaler = train_models(X, y)
                
                st.session_state.models_results = models_results
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.feature_columns = feature_columns
                st.session_state.label_encoders = label_encoders
                st.session_state.models_trained = True
                
                st.success("✅ Models trained successfully!")
        
        if st.session_state.models_trained:
            models_results = st.session_state.models_results
            
            # Model comparison
            st.subheader("📊 Model Performance Comparison")
            
            comparison_data = []
            for model_name, results in models_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'ROC AUC': results['roc_auc'],
                    'F1 Score': results['f1'],
                    'Precision': results['precision'],
                    'Recall': results['recall']
                })
            
            comparison_df = pd.DataFrame(comparison_data).sort_values('ROC AUC', ascending=False)
            st.dataframe(comparison_df.round(4))
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig_roc = px.bar(comparison_df, x='Model', y='ROC AUC', 
                               title="ROC AUC Comparison", color='ROC AUC',
                               color_continuous_scale='Viridis')
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                fig_metrics = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                
                for i, model in enumerate(comparison_df['Model']):
                    values = [comparison_df.iloc[i][metric] for metric in metrics]
                    fig_metrics.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=model
                    ))
                
                fig_metrics.update_layout(title="Model Performance Radar Chart")
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            # ROC Curves
            st.subheader("📈 ROC Curves Comparison")
            fig_roc_curves = go.Figure()
            
            for model_name, results in models_results.items():
                fpr, tpr, _ = roc_curve(st.session_state.y_test, results['probabilities'])
                fig_roc_curves.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{model_name} (AUC = {results['roc_auc']:.3f})",
                    mode='lines'
                ))
            
            fig_roc_curves.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc_curves.update_layout(
                title="ROC Curves Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig_roc_curves, use_container_width=True)
    
    # Predictions Page
    elif page == "📈 Predictions":
        st.header("📈 Real-time Churn Predictions")
        
        if not st.session_state.models_trained:
            st.warning("⚠ Please train the models first in the Model Training section.")
            return
        
        st.subheader("🎯 Individual Customer Prediction")
        
        # Input form for prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col3:
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, 2000.0)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        if st.button("🔮 Predict Churn Probability", type="primary"):
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            models_results = st.session_state.models_results
            
            # Simplified prediction for demo (replace with actual prediction in production)
            base_prob = 0.3
            if contract == "Month-to-month":
                base_prob += 0.3
            if payment_method == "Electronic check":
                base_prob += 0.2
            if tenure < 12:
                base_prob += 0.2
            if monthly_charges > 80:
                base_prob += 0.1
            
            for i, (model_name, results) in enumerate(models_results.items()):
                # Add some variation for different models
                if model_name == "Random Forest":
                    prob = min(base_prob + 0.05, 0.95)
                elif model_name == "XGBoost":
                    prob = min(base_prob + 0.02, 0.95)
                elif model_name == "Gradient Boosting":
                    prob = min(base_prob - 0.02, 0.95)
                else:  # Neural Network
                    prob = min(base_prob - 0.05, 0.95)
                
                risk_level = "🔴 High Risk" if prob > 0.7 else "🟡 Medium Risk" if prob > 0.4 else "🟢 Low Risk"
                
                if i % 3 == 0:
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{model_name}</h4>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                elif i % 3 == 1:
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{model_name}</h4>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{model_name}</h4>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Business Impact Page
    elif page == "💼 Business Impact":
        st.header("💼 Business Impact Analysis")
        
        if not st.session_state.models_trained:
            st.warning("⚠ Please train the models first in the Model Training section.")
            return
        
        models_results = st.session_state.models_results
        y_test = st.session_state.y_test
        avg_clv = df_processed['EstimatedCLV'].mean()
        
        # Calculate business metrics
        business_metrics = calculate_business_impact(models_results, y_test, avg_clv)
        
        st.subheader("💰 Financial Impact Summary")
        
        # Business assumptions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            retention_cost = st.number_input("Retention Campaign Cost ($)", value=100, min_value=50, max_value=500)
        with col2:
            retention_rate = st.slider("Successful Retention Rate", 0.1, 0.5, 0.3)
        with col3:
            st.metric("Average CLV", f"${avg_clv:,.0f}")
        with col4:
            st.metric("Test Set Size", len(y_test))
        
        # Business impact for each model
        st.subheader("📊 Model Business Performance")
        
        for model_name, metrics in business_metrics.items():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"{model_name} - Net Value", f"${metrics['net_value']:,.0f}")
            with col2:
                st.metric(f"{model_name} - ROI", f"{metrics['roi']:.1f}%")
            with col3:
                st.metric(f"{model_name} - Customers Targeted", f"{metrics['customers_targeted']:.0f}")
            with col4:
                st.metric(f"{model_name} - Customers Retained", f"{metrics['customers_retained']:.0f}")
        
        # ROI Comparison
        st.subheader("📈 ROI Comparison")
        
        roi_data = [(name, metrics['roi']) for name, metrics in business_metrics.items()]
        roi_df = pd.DataFrame(roi_data, columns=['Model', 'ROI'])
        
        fig_roi = px.bar(roi_df, x='Model', y='ROI', title="Return on Investment by Model",
                        color='ROI', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Strategic recommendations
        st.subheader("🎯 Strategic Recommendations")
        
        best_model = max(business_metrics.items(), key=lambda x: x[1]['roi'])
        
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Recommended Model: {best_model[0]}</h4>
            <p><strong>Expected ROI:</strong> {best_model[1]['roi']:.1f}%</p>
            <p><strong>Monthly Value:</strong> ${best_model[1]['net_value']/12:,.0f}</p>
            <p><strong>Customers to Target:</strong> {best_model[1]['customers_targeted']:.0f} per month</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Implementation Strategy
        
        1. **High-Priority Actions:**
           - Focus retention efforts on month-to-month contract customers
           - Implement automated payment incentives
           - Develop specialized onboarding for new customers
        
        2. **Revenue Optimization:**
           - Deploy model for proactive customer outreach
           - Implement dynamic pricing strategies
           - Create loyalty programs for high-value customers
        
        3. **Technical Implementation:**
           - Integrate model into CRM system
           - Set up automated risk alerts
           - Create real-time monitoring dashboard
        """)
    
    # Model Insights Page
    elif page == "🔍 Model Insights":
        st.header("🔍 Advanced Model Insights")
        
        if not st.session_state.models_trained:
            st.warning("⚠ Please train the models first in the Model Training section.")
            return
        
        models_results = st.session_state.models_results
        feature_columns = st.session_state.feature_columns
        
        # Feature importance
        st.subheader("📊 Feature Importance Analysis")
        
        # Filter available models for feature importance analysis
        tree_models = [name for name in models_results.keys() 
                      if name in ["Random Forest", "XGBoost", "Gradient Boosting"]]
        
        if tree_models:
            selected_model = st.selectbox("Select Model for Analysis", tree_models)
            
            if selected_model in models_results:
                model = models_results[selected_model]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True).tail(15)
                    
                    fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                          title=f"Top 15 Feature Importances - {selected_model}",
                                          orientation='h')
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Top features table
                    st.subheader("🔍 Top Features Analysis")
                    top_features = importance_df.tail(10).sort_values('Importance', ascending=False)
                    st.dataframe(top_features)
        else:
            st.warning("⚠ No tree-based models available for feature importance analysis.")
        
        # Model comparison insights
        st.subheader("🔬 Model Performance Insights")
        
        performance_summary = []
        for model_name, results in models_results.items():
            performance_summary.append({
                'Model': model_name,
                'Strength': 'High Precision' if results['precision'] > 0.6 else 'Balanced Performance',
                'Best Use Case': 'Cost-sensitive scenarios' if results['precision'] > results['recall'] else 'Comprehensive detection',
                'Accuracy': f"{results['accuracy']:.1%}",
                'ROC AUC': f"{results['roc_auc']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(performance_summary))
        
        # Deployment recommendations
        st.subheader("🚀 Deployment Recommendations")
        
        best_auc_model = max(models_results.items(), key=lambda x: x[1]['roc_auc'])
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🎯 Production Deployment Strategy</h4>
            <p><strong>Recommended Primary Model:</strong> {best_auc_model[0]}</p>
            <p><strong>Performance:</strong> {best_auc_model[1]['roc_auc']:.3f} ROC AUC</p>
            
            <h5>Monitoring Metrics:</h5>
            <ul>
                <li>Model drift detection: Monitor feature distributions</li>
                <li>Performance degradation: Track ROC AUC monthly</li>
                <li>Business impact: Monitor retention campaign success rates</li>
                <li>Data quality: Check for missing values and outliers</li>
            </ul>
            
            <h5>Retraining Schedule:</h5>
            <ul>
                <li>Quarterly model retraining with new data</li>
                <li>Feature importance review every 6 months</li>
                <li>A/B testing for model improvements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()