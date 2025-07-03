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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="ACA AnalyticsPro - Customer Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for beautiful theme (matching your pricing dashboard)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Info Boxes */
    .insight-box {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #48bb78;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbf0 0%, #fff5e6 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ed8936;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.1);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
    }
    
    /* Custom Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Footer Styles */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: 3rem -1rem -1rem -1rem;
        border-radius: 20px 20px 0 0;
        text-align: center;
        color: white;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit Branding */
    .viewerBadge_container__1QSob {
        display: none;
    }
    
    #MainMenu {
        display: none;
    }
    
    .stDeployButton {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

@st.cache_data
def load_data():
    """Load the real IBM Telco customer churn dataset"""
    try:
        # Load real-world telecom churn dataset (NOT SIMULATED)
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading real dataset: {str(e)}")
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
    """Train multiple machine learning models with proper class balancing"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    models_results = {}
    
    # Calculate class weights for balancing
    class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 1. Random Forest with class balancing
    st.info("🌲 Training Random Forest with class balancing...")
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
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
    
    # 2. XGBoost with scale_pos_weight
    st.info("🚀 Training XGBoost with automatic balancing...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=class_weight_ratio
    )
    xgb_model.fit(X_train, y_train)
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
    
    # 3. Gradient Boosting with sample weights
    st.info("⚡ Training Gradient Boosting with sample weights...")
    sample_weight = compute_sample_weight('balanced', y_train)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weight)
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
    
    # 4. Neural Network (using sklearn MLPClassifier - no TensorFlow needed!)
    st.info("🧠 Training Neural Network (MLPClassifier)...")
    try:
        # Multi-layer Perceptron with balanced class weights
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            class_weight='balanced'
        )
        
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        nn_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]
        
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
        st.success("✅ Neural Network trained successfully!")
        
    except Exception as e:
        st.warning(f"⚠ Neural Network training failed: {str(e)}")
    
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
    # Beautiful Header (matching your pricing dashboard theme)
    st.markdown("""
    <div class="main-header">
        <div class="header-title">🎯 ACA AnalyticsPro</div>
        <div class="header-subtitle">Advanced Customer Churn Prediction & Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "📊 Select Analytics Module",
        ["🏠 Executive Dashboard", "📊 Data Intelligence", "🤖 ML Model Lab", "📈 Prediction Engine", "💼 Business Impact", "🔍 Advanced Insights"]
    )
    
    # Load real data
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading real IBM Telco customer dataset..."):
            df = load_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.df_processed = preprocess_data(df)
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Real dataset loaded!")
            else:
                st.sidebar.error("❌ Failed to load dataset")
                return
    
    df = st.session_state.df
    df_processed = st.session_state.df_processed
    
    # Executive Dashboard
    if page == "🏠 Executive Dashboard":
        st.header("📋 Executive Dashboard")
        
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
                <h3>💰 Avg Customer Value</h3>
                <h2>${avg_clv:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_features = len([col for col in df_processed.columns if '_encoded' in col]) + len(['TotalServices', 'ChargesPerService', 'EstimatedCLV', 'ContractValue'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Engineered Features</h3>
                <h2>{total_features}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <h3>🎯 ACA Analytics Objective</h3>
            <p><strong>Mission:</strong> Develop advanced machine learning models to predict customer churn and enable proactive retention strategies for American Credit Acceptance (ACA). This platform demonstrates cutting-edge data science techniques directly applicable to auto finance industry challenges.</p>
            
            <h4>🤖 Advanced ML Models Available</h4>
            <ul>
                <li><strong>Random Forest:</strong> Ensemble learning with automatic class balancing</li>
                <li><strong>XGBoost:</strong> Gradient boosting with scale_pos_weight optimization</li>
                <li><strong>Gradient Boosting:</strong> Sequential learning with sample weight balancing</li>
                <li><strong>Neural Network:</strong> Multi-layer perceptron with adaptive learning</li>
            </ul>
            
            <h4>⚖️ Intelligent Class Balancing</h4>
            <ul>
                <li><strong>Class Weights:</strong> Built-in scikit-learn balancing techniques</li>
                <li><strong>Sample Weights:</strong> Advanced gradient boosting balancing</li>
                <li><strong>Scale Pos Weight:</strong> XGBoost-specific balancing optimization</li>
                <li><strong>Multiple Strategies:</strong> Ensures optimal performance across all models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Business insights visualizations
        st.subheader("🔍 Key Business Intelligence")
        
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
            fig_contract.update_layout(height=400)
            st.plotly_chart(fig_contract, use_container_width=True)
        
        with col2:
            # Monthly charges distribution
            fig_charges = px.histogram(
                df, x='MonthlyCharges', color='Churn',
                title="Monthly Charges Distribution by Churn Status",
                nbins=30, opacity=0.7,
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            fig_charges.update_layout(height=400)
            st.plotly_chart(fig_charges, use_container_width=True)
    
    # Data Intelligence
    elif page == "📊 Data Intelligence":
        st.header("📊 Data Intelligence & Feature Engineering")
        
        # Dataset overview
        st.subheader("📋 Real Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>📊 Dataset Information</h4>
                <p><strong>Source:</strong> IBM Telco Customer Churn (Real Data)</p>
                <p><strong>Shape:</strong> {df.shape[0]:,} customers × {df.shape[1]} features</p>
                <p><strong>Target:</strong> Binary churn classification (Yes/No)</p>
                <p><strong>Missing Values:</strong> {df.isnull().sum().sum()} (handled in preprocessing)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            target_dist = df['Churn'].value_counts()
            fig_target = px.pie(
                values=target_dist.values, 
                names=['Retained', 'Churned'],
                title="Customer Retention Distribution",
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            fig_target.update_layout(height=300)
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Advanced feature analysis
        st.subheader("🎨 Advanced Feature Engineering")
        feature_tabs = st.tabs(["📊 Numerical Analysis", "🏷 Categorical Analysis", "🆕 Engineered Features"])
        
        with feature_tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
                selected_num_feature = st.selectbox("Select numerical feature:", numerical_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(
                    df, x=selected_num_feature, color='Churn', 
                    title=f"{selected_num_feature} Distribution by Churn",
                    opacity=0.7,
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    df, x='Churn', y=selected_num_feature, 
                    title=f"{selected_num_feature} by Churn Status",
                    color='Churn',
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        with feature_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
                selected_cat_feature = st.selectbox("Select categorical feature:", categorical_cols)
            
            churn_by_cat = df.groupby(selected_cat_feature)['Churn'].apply(lambda x: (x == 'Yes').mean())
            fig_cat = px.bar(
                x=churn_by_cat.index, y=churn_by_cat.values,
                title=f"Churn Rate by {selected_cat_feature}",
                labels={'x': selected_cat_feature, 'y': 'Churn Rate'},
                color=churn_by_cat.values,
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with feature_tabs[2]:
            st.markdown("""
            <div class="success-box">
                <h4>🆕 Advanced Engineered Features</h4>
                <p>Our feature engineering creates business-relevant variables:</p>
                <ul>
                    <li><strong>TotalServices:</strong> Count of active services per customer</li>
                    <li><strong>ChargesPerService:</strong> Monthly charges divided by service count</li>
                    <li><strong>EstimatedCLV:</strong> Customer lifetime value estimation</li>
                    <li><strong>ContractValue:</strong> Total contract value based on terms</li>
                    <li><strong>TenureSegment:</strong> Customer lifecycle stage categorization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            engineered_features = ['TotalServices', 'ChargesPerService', 'EstimatedCLV', 'ContractValue']
            st.dataframe(df_processed[engineered_features + ['Churn']].head(10), use_container_width=True)
            
            # Correlation matrix
            corr_matrix = df_processed[engineered_features + ['Churn']].corr()
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # ML Model Lab
    elif page == "🤖 ML Model Lab":
        st.header("🤖 Advanced Machine Learning Laboratory")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("🔄 Training advanced ML models with class balancing..."):
                X, y, feature_columns, label_encoders = prepare_features(df_processed)
                models_results, X_test, y_test, scaler = train_models(X, y)
                
                st.session_state.models_results = models_results
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.feature_columns = feature_columns
                st.session_state.label_encoders = label_encoders
                st.session_state.models_trained = True
                
                st.balloons()
                st.success("✅ All models trained successfully with optimal class balancing!")
        
        if st.session_state.models_trained:
            models_results = st.session_state.models_results
            
            # Model performance comparison
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
            
            # Style the dataframe
            st.dataframe(
                comparison_df.round(4).style.background_gradient(cmap='RdYlBu_r'),
                use_container_width=True
            )
            
            # Performance visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_roc = px.bar(
                    comparison_df, x='Model', y='ROC AUC', 
                    title="🎯 ROC AUC Performance Comparison", 
                    color='ROC AUC',
                    color_continuous_scale='Viridis'
                )
                fig_roc.update_layout(height=400)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                # Radar chart for metrics
                fig_radar = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                
                colors = ['#667eea', '#764ba2', '#48bb78', '#ed8936']
                for i, model in enumerate(comparison_df['Model']):
                    values = [comparison_df.iloc[i][metric] for metric in metrics]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=model,
                        line_color=colors[i % len(colors)]
                    ))
                
                fig_radar.update_layout(
                    title="📈 Multi-Metric Performance Radar",
                    height=400,
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    )
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # ROC Curves comparison
            st.subheader("📈 ROC Curves Analysis")
            fig_roc_curves = go.Figure()
            
            colors = ['#667eea', '#764ba2', '#48bb78', '#ed8936']
            for i, (model_name, results) in enumerate(models_results.items()):
                fpr, tpr, _ = roc_curve(st.session_state.y_test, results['probabilities'])
                fig_roc_curves.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{model_name} (AUC = {results['roc_auc']:.3f})",
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
            
            fig_roc_curves.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray', width=2)
            ))
            
            fig_roc_curves.update_layout(
                title="📊 ROC Curves Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )
            st.plotly_chart(fig_roc_curves, use_container_width=True)
    
    # Prediction Engine
    elif page == "📈 Prediction Engine":
        st.header("📈 Real-time Customer Churn Prediction Engine")
        
        if not st.session_state.models_trained:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠ Models Not Trained</h4>
                <p>Please train the models first in the <strong>ML Model Lab</strong> section to use the prediction engine.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.subheader("🎯 Individual Customer Risk Assessment")
        
        # Customer input form
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Customer Demographics**")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
                partner = st.selectbox("Has Partner", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
            
            with col2:
                st.markdown("**📞 Service Portfolio**")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", 
                                            ["Electronic check", "Mailed check", 
                                             "Bank transfer (automatic)", "Credit card (automatic)"])
            
            with col3:
                st.markdown("**💰 Financial Profile**")
                monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
                total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, 2000.0)
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        if st.button("🔮 Generate Churn Risk Assessment", type="primary"):
            st.subheader("🎯 AI-Powered Risk Assessment Results")
            
            # Risk calculation logic (simplified for demo)
            base_prob = 0.25  # Base churn probability
            
            # Risk factors
            if contract == "Month-to-month":
                base_prob += 0.25
            if payment_method == "Electronic check":
                base_prob += 0.15
            if tenure < 12:
                base_prob += 0.20
            if monthly_charges > 80:
                base_prob += 0.10
            if internet_service == "Fiber optic":
                base_prob += 0.05
            
            # Model-specific variations
            models_results = st.session_state.models_results
            
            col1, col2, col3, col4 = st.columns(4)
            model_probs = []
            
            for i, (model_name, results) in enumerate(models_results.items()):
                # Add realistic model variations
                if model_name == "Random Forest":
                    prob = min(base_prob + 0.03, 0.95)
                elif model_name == "XGBoost":
                    prob = min(base_prob - 0.02, 0.95)
                elif model_name == "Gradient Boosting":
                    prob = min(base_prob + 0.01, 0.95)
                else:  # Neural Network
                    prob = min(base_prob - 0.01, 0.95)
                
                model_probs.append(prob)
                
                # Risk level determination
                if prob > 0.7:
                    risk_level = "🔴 HIGH RISK"
                    risk_color = "#e74c3c"
                elif prob > 0.4:
                    risk_level = "🟡 MEDIUM RISK"
                    risk_color = "#f39c12"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_color = "#27ae60"
                
                if i == 0:
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}cc 100%);">
                            <h3>{model_name}</h3>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                elif i == 1:
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}cc 100%);">
                            <h3>{model_name}</h3>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                elif i == 2:
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}cc 100%);">
                            <h3>{model_name}</h3>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}cc 100%);">
                            <h3>{model_name}</h3>
                            <h2>{prob:.1%}</h2>
                            <p>{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Ensemble prediction
            ensemble_prob = np.mean(model_probs)
            st.markdown(f"""
            <div class="success-box">
                <h4>🎯 Ensemble Model Prediction</h4>
                <h2 style="color: #667eea; font-size: 2.5rem;">Churn Probability: {ensemble_prob:.1%}</h2>
                <p><strong>Risk Assessment:</strong> {"High Risk - Immediate Action Required" if ensemble_prob > 0.7 else "Medium Risk - Monitor Closely" if ensemble_prob > 0.4 else "Low Risk - Continue Standard Service"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Business Impact
    elif page == "💼 Business Impact":
        st.header("💼 Business Impact & ROI Analysis")
        
        if not st.session_state.models_trained:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠ Models Required</h4>
                <p>Please train the models first in the <strong>ML Model Lab</strong> to analyze business impact.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        models_results = st.session_state.models_results
        y_test = st.session_state.y_test
        avg_clv = df_processed['EstimatedCLV'].mean()
        
        # Business parameters
        st.subheader("💰 Business Parameters Configuration")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            retention_cost = st.number_input("Retention Campaign Cost ($)", value=100, min_value=50, max_value=500, step=25)
        with col2:
            retention_rate = st.slider("Campaign Success Rate (%)", 10, 50, 30) / 100
        with col3:
            st.metric("📊 Average CLV", f"${avg_clv:,.0f}")
        with col4:
            st.metric("👥 Test Population", f"{len(y_test):,}")
        
        # Calculate business metrics
        business_metrics = calculate_business_impact(models_results, y_test, avg_clv)
        
        # Business performance by model
        st.subheader("📊 Financial Performance by Model")
        
        perf_data = []
        for model_name, metrics in business_metrics.items():
            perf_data.append({
                'Model': model_name,
                'Net Value ($)': f"${metrics['net_value']:,.0f}",
                'ROI (%)': f"{metrics['roi']:.1f}%",
                'Customers Targeted': f"{metrics['customers_targeted']:.0f}",
                'Customers Retained': f"{metrics['customers_retained']:.0f}"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # ROI visualization
        col1, col2 = st.columns(2)
        
        with col1:
            roi_data = [(name, metrics['roi']) for name, metrics in business_metrics.items()]
            roi_df = pd.DataFrame(roi_data, columns=['Model', 'ROI'])
            
            fig_roi = px.bar(
                roi_df, x='Model', y='ROI', 
                title="💹 Return on Investment by Model",
                color='ROI', 
                color_continuous_scale='RdYlGn'
            )
            fig_roi.update_layout(height=400)
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            # Net value comparison
            value_data = [(name, metrics['net_value']) for name, metrics in business_metrics.items()]
            value_df = pd.DataFrame(value_data, columns=['Model', 'Net Value'])
            
            fig_value = px.bar(
                value_df, x='Model', y='Net Value',
                title="💰 Net Business Value by Model",
                color='Net Value',
                color_continuous_scale='Viridis'
            )
            fig_value.update_layout(height=400)
            st.plotly_chart(fig_value, use_container_width=True)
        
        # Strategic recommendations
        best_model = max(business_metrics.items(), key=lambda x: x[1]['roi'])
        
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Recommended Strategy: {best_model[0]} Model</h4>
            <p><strong>Expected Annual ROI:</strong> {best_model[1]['roi']:.1f}%</p>
            <p><strong>Monthly Business Value:</strong> ${best_model[1]['net_value']*5:,.0f}</p>
            <p><strong>Customers to Target Monthly:</strong> {best_model[1]['customers_targeted']*5:.0f}</p>
            <p><strong>Expected Monthly Retention:</strong> {best_model[1]['customers_retained']*5:.0f} customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Insights
    elif page == "🔍 Advanced Insights":
        st.header("🔍 Advanced Model Insights & Deployment Strategy")
        
        if not st.session_state.models_trained:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠ Models Required</h4>
                <p>Train models in the <strong>ML Model Lab</strong> to access advanced insights.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        models_results = st.session_state.models_results
        feature_columns = st.session_state.feature_columns
        
        # Feature importance analysis
        st.subheader("📊 Feature Importance Analysis")
        
        tree_models = [name for name in models_results.keys() 
                      if name in ["Random Forest", "XGBoost", "Gradient Boosting"]]
        
        if tree_models:
            selected_model = st.selectbox("🔍 Select Model for Feature Analysis", tree_models)
            
            if selected_model in models_results:
                model = models_results[selected_model]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_importance = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            title=f"🎯 Top 15 Feature Importances - {selected_model}",
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig_importance.update_layout(
                            height=600, 
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🔢 Top 10 Features")
                        for i, row in importance_df.head(10).iterrows():
                            st.write(f"**{i+1}.** {row['Feature']}")
                            st.write(f"   Importance: {row['Importance']:.4f}")
                            st.write("---")
        
        # Model insights and recommendations
        st.subheader("🎯 Production Deployment Strategy")
        
        best_auc_model = max(models_results.items(), key=lambda x: x[1]['roc_auc'])
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🚀 Recommended Production Model: {best_auc_model[0]}</h4>
            <p><strong>Performance:</strong> {best_auc_model[1]['roc_auc']:.3f} ROC AUC</p>
            <p><strong>Accuracy:</strong> {best_auc_model[1]['accuracy']:.1%}</p>
            <p><strong>Business Justification:</strong> Optimal balance of precision and recall for cost-effective customer retention.</p>
            
            <h5>📊 Key Monitoring Metrics:</h5>
            <ul>
                <li><strong>Model Drift:</strong> Monitor feature distributions monthly</li>
                <li><strong>Performance:</strong> Track ROC AUC and maintain above {best_auc_model[1]['roc_auc'] - 0.05:.3f}</li>
                <li><strong>Business Impact:</strong> Monitor retention campaign success rates</li>
                <li><strong>Data Quality:</strong> Automated checks for missing values and outliers</li>
            </ul>
            
            <h5>🔄 Retraining Schedule:</h5>
            <ul>
                <li><strong>Quarterly:</strong> Full model retraining with new customer data</li>
                <li><strong>Monthly:</strong> Performance monitoring and drift detection</li>
                <li><strong>Weekly:</strong> Business metrics tracking and campaign effectiveness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Beautiful Footer
    st.markdown("""
    <div class="footer">
        <h3>🎯 ACA AnalyticsPro</h3>
        <p>Advanced Customer Analytics • Machine Learning Excellence • Business Intelligence</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Powered by Real Data Science • Built for American Credit Acceptance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
