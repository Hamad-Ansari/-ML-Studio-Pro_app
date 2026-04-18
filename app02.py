import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
import time
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Regression models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ===============================================
# 🎨 PAGE CONFIGURATION
# ===============================================
st.set_page_config(
    page_title="ML Studio Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================
# 🎨 CUSTOM CSS STYLING
# ===============================================
def load_custom_css():
    st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
        100% { filter: hue-rotate(0deg); }
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hero section */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        animation: slideDown 1s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.3rem;
        margin-bottom: 30px;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Best model highlight */
    .best-model {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 30px rgba(245, 87, 108, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error messages */
    .success-msg {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }
    
    .error-msg {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    </style>
    """, unsafe_allow_html=True)

# ===============================================
# 🔧 UTILITY FUNCTIONS
# ===============================================

@st.cache_data
def load_example_dataset(dataset_name):
    """Load example datasets from seaborn"""
    try:
        df = sns.load_dataset(dataset_name)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_uploaded_file(uploaded_file):
    """Load uploaded file based on extension"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'tsv':
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error("Unsupported file format")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(X, y, test_size=0.2):
    """
    Comprehensive preprocessing pipeline:
    - Handle missing values
    - Encode categorical variables
    - Scale features
    - Train-test split
    """
    
    # Store original column names
    feature_columns = X.columns.tolist()
    
    # Handle categorical columns
    label_encoders = {}
    X_processed = X.copy()
    
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
    
    # Encode target if categorical
    y_encoder = None
    if y.dtype == 'object':
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
    
    # Handle missing values with IterativeImputer
    imputer = IterativeImputer(random_state=42, max_iter=10)
    X_imputed = imputer.fit_transform(X_processed)
    X_imputed = pd.DataFrame(X_imputed, columns=feature_columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'imputer': imputer,
        'label_encoders': label_encoders,
        'y_encoder': y_encoder,
        'feature_names': feature_columns
    }

def get_regression_models():
    """Return dictionary of regression models"""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
        'Support Vector Regressor': SVR()
    }

def get_classification_models():
    """Return dictionary of classification models"""
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=100),
        'Support Vector Classifier': SVC(random_state=42)
    }

def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R² Score': r2_score(y_true, y_pred)
    }

def evaluate_classification(y_true, y_pred):
    """Calculate classification metrics"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create interactive confusion matrix with Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Pred {i}' for i in range(len(cm))],
        y=[f'True {i}' for i in range(len(cm))],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_dark',
        height=400
    )
    
    return fig

def plot_metrics_comparison(results, problem_type):
    """Create interactive bar chart comparing model metrics"""
    
    if problem_type == 'Regression':
        metric_key = 'R² Score'
    else:
        metric_key = 'F1 Score'
    
    models = list(results.keys())
    scores = [results[model][metric_key] for model in models]
    
    colors = ['#667eea' if score != max(scores) else '#f5576c' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=scores,
            marker_color=colors,
            text=[f'{score:.4f}' for score in scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' + metric_key + ': %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'Model Comparison - {metric_key}',
        xaxis_title='Model',
        yaxis_title=metric_key,
        template='plotly_dark',
        height=500,
        showlegend=False
    )
    
    return fig

def create_metric_card(label, value, icon="📊"):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================================
# 🚀 MAIN APPLICATION
# ===============================================

def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    
    # ===============================================
    # 🌟 HERO SECTION
    # ===============================================
    
    st.markdown('<h1 class="hero-title">🤖 ML Studio Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Production-Grade Machine Learning Platform with Beautiful UI</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===============================================
    # 📂 SIDEBAR - DATA SOURCE SELECTION
    # ===============================================
    
    with st.sidebar:
        st.markdown("## 🎯 Configuration Panel")
        st.markdown("---")
        
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Upload Dataset", "Example Dataset"],
            label_visibility="collapsed"
        )
        
        df = None
        
        if data_source == "Upload Dataset":
            st.markdown("#### 📤 Upload File")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'tsv'],
                help="Supported formats: CSV, XLSX, TSV"
            )
            
            if uploaded_file is not None:
                df = load_uploaded_file(uploaded_file)
                if df is not None:
                    st.session_state.data_loaded = True
                    st.success("✅ File loaded successfully!")
        
        else:
            st.markdown("#### 📊 Example Datasets")
            dataset_name = st.selectbox(
                "Select dataset:",
                ["titanic", "tips", "iris", "penguins", "diamonds"]
            )
            
            if st.button("🔄 Load Dataset", use_container_width=True):
                with st.spinner("Loading dataset..."):
                    df = load_example_dataset(dataset_name)
                    if df is not None:
                        st.session_state.data_loaded = True
                        st.success("✅ Dataset loaded!")
        
        st.markdown("---")
        
        # Store dataframe in session state
        if df is not None:
            st.session_state.df = df
    
    # ===============================================
    # 📊 MAIN CONTENT AREA
    # ===============================================
    
    if not st.session_state.data_loaded or 'df' not in st.session_state:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h2 style="text-align: center; color: white;">👋 Welcome to ML Studio Pro</h2>
                <p style="color: white; text-align: center; font-size: 1.1rem;">
                Your all-in-one platform for machine learning experiments
                </p>
                <br>
                <h3 style="color: white;">✨ Features:</h3>
                <ul style="color: white; font-size: 1.1rem;">
                    <li>🎨 Modern, beautiful UI with smooth animations</li>
                    <li>📊 Interactive data exploration</li>
                    <li>🤖 Multiple ML algorithms (Regression & Classification)</li>
                    <li>📈 Comprehensive model evaluation</li>
                    <li>💾 Model export and deployment</li>
                    <li>🔮 Real-time predictions</li>
                </ul>
                <br>
                <p style="color: white; text-align: center; font-size: 1.2rem; font-weight: 600;">
                👈 Get started by selecting a data source from the sidebar!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem;">🎨</div>
                <div class="metric-label">Modern UI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem;">⚡</div>
                <div class="metric-label">Fast Processing</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem;">🎯</div>
                <div class="metric-label">High Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 3rem;">📊</div>
                <div class="metric-label">Rich Analytics</div>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # ===============================================
    # 🎯 TABBED INTERFACE
    # ===============================================
    
    tabs = st.tabs(["📊 Data Overview", "⚙️ Configuration", "🎯 Training & Results", "🔮 Predictions"])
    
    df = st.session_state.df
    
    # ===============================================
    # TAB 1: DATA OVERVIEW
    # ===============================================
    
    with tabs[0]:
        st.markdown("## 📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Rows", f"{df.shape[0]:,}", "📏")
        with col2:
            create_metric_card("Columns", f"{df.shape[1]:,}", "📐")
        with col3:
            create_metric_card("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB", "💾")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            create_metric_card("Missing", f"{missing_pct:.1f}%", "🔍")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data preview
        with st.expander("🔍 Data Preview (First 10 Rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📋 Column Information"):
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True)
        
        with col2:
            with st.expander("📊 Statistical Summary"):
                st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values visualization
        if df.isnull().sum().sum() > 0:
            st.markdown("### 🔍 Missing Values Analysis")
            
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = go.Figure(data=[
                    go.Bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        marker_color='#f5576c',
                        text=missing_data.values,
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title='Missing Values by Column',
                    xaxis_title='Column',
                    yaxis_title='Missing Count',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ===============================================
    # TAB 2: CONFIGURATION
    # ===============================================
    
    with tabs[1]:
        st.markdown("## ⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Feature Selection")
            
            # Get numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            all_cols = df.columns.tolist()
            
            feature_columns = st.multiselect(
                "Select Feature Columns (X):",
                options=all_cols,
                default=all_cols[:min(5, len(all_cols))],
                help="Choose the columns to use as features for training"
            )
            
            target_column = st.selectbox(
                "Select Target Column (y):",
                options=[col for col in all_cols if col not in feature_columns],
                help="Choose the column to predict"
            )
            
            st.markdown("---")
            
            st.markdown("### 🔬 Problem Type")
            problem_type = st.radio(
                "Select the type of problem:",
                ["Regression", "Classification"],
                help="Choose based on your target variable type"
            )
            
            st.session_state.problem_type = problem_type
        
        with col2:
            st.markdown("### 🔀 Train-Test Split")
            
            test_size = st.slider(
                "Test Set Size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            ) / 100
            
            st.info(f"""
            **Split Configuration:**
            - Training Set: {(1-test_size)*100:.0f}% ({int(df.shape[0]*(1-test_size)):,} samples)
            - Test Set: {test_size*100:.0f}% ({int(df.shape[0]*test_size):,} samples)
            """)
            
            st.markdown("---")
            
            st.markdown("### 🤖 Model Selection")
            
            if problem_type == "Regression":
                models_dict = get_regression_models()
            else:
                models_dict = get_classification_models()
            
            selected_models = st.multiselect(
                "Choose models to train:",
                options=list(models_dict.keys()),
                default=list(models_dict.keys()),
                help="Select one or more models to train and compare"
            )
        
        st.markdown("---")
        
        # Run Analysis Button
        st.markdown("### ▶️ Start Analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_analysis = st.button(
                "🚀 Run Preprocessing & Analysis",
                use_container_width=True,
                type="primary"
            )
        
        if run_analysis:
            if not feature_columns:
                st.error("❌ Please select at least one feature column!")
                return
            
            if not target_column:
                st.error("❌ Please select a target column!")
                return
            
            if not selected_models:
                st.error("❌ Please select at least one model!")
                return
            
            with st.spinner("🔄 Preprocessing data..."):
                try:
                    X = df[feature_columns]
                    y = df[target_column]
                    
                    # Preprocess data
                    preprocessing_results = preprocess_data(X, y, test_size)
                    
                    # Store in session state
                    st.session_state.preprocessing_results = preprocessing_results
                    st.session_state.selected_models = selected_models
                    st.session_state.models_dict = models_dict
                    st.session_state.feature_columns = feature_columns
                    st.session_state.target_column = target_column
                    st.session_state.preprocessing_done = True
                    
                    st.success("✅ Preprocessing completed successfully!")
                    
                    # Show preprocessing summary
                    st.markdown("#### 📋 Preprocessing Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        create_metric_card(
                            "Features",
                            len(feature_columns),
                            "🎯"
                        )
                    
                    with col2:
                        create_metric_card(
                            "Training Samples",
                            len(preprocessing_results['y_train']),
                            "📚"
                        )
                    
                    with col3:
                        create_metric_card(
                            "Test Samples",
                            len(preprocessing_results['y_test']),
                            "🧪"
                        )
                    
                    st.info("✅ Data has been preprocessed with:\n- Missing value imputation\n- Feature scaling\n- Categorical encoding")
                    
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")
    
    # ===============================================
    # TAB 3: TRAINING & RESULTS
    # ===============================================
    
    with tabs[2]:
        st.markdown("## 🎯 Model Training & Evaluation")
        
        if not st.session_state.preprocessing_done:
            st.warning("⚠️ Please complete the configuration and run analysis first (Tab 2)")
            return
        
        # Train Models Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            train_button = st.button(
                "🎯 Train Models",
                use_container_width=True,
                type="primary"
            )
        
        if train_button:
            preprocessing_results = st.session_state.preprocessing_results
            selected_models = st.session_state.selected_models
            models_dict = st.session_state.models_dict
            problem_type = st.session_state.problem_type
            
            X_train = preprocessing_results['X_train']
            X_test = preprocessing_results['X_test']
            y_train = preprocessing_results['y_train']
            y_test = preprocessing_results['y_test']
            
            results = {}
            trained_models = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}... ({idx+1}/{len(selected_models)})")
                
                model = models_dict[model_name]
                
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate
                if problem_type == "Regression":
                    metrics = evaluate_regression(y_test, y_pred)
                else:
                    metrics = evaluate_classification(y_test, y_pred)
                
                metrics['Training Time (s)'] = training_time
                
                results[model_name] = metrics
                trained_models[model_name] = {
                    'model': model,
                    'predictions': y_pred
                }
                
                progress_bar.progress((idx + 1) / len(selected_models))
            
            status_text.text("✅ Training completed!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Store results
            st.session_state.results = results
            st.session_state.trained_models = trained_models
            st.session_state.models_trained = True
            
            st.success("🎉 All models trained successfully!")
        
        # Display Results
        if st.session_state.models_trained:
            results = st.session_state.results
            trained_models = st.session_state.trained_models
            problem_type = st.session_state.problem_type
            
            st.markdown("---")
            st.markdown("### 📊 Model Performance Comparison")
            
            # Metrics comparison chart
            fig = plot_metrics_comparison(results, problem_type)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed metrics table
            st.markdown("### 📋 Detailed Metrics")
            
            results_df = pd.DataFrame(results).T
            results_df = results_df.round(4)
            
            # Highlight best values
            st.dataframe(
                results_df.style.highlight_max(axis=0, color='lightgreen') if problem_type == "Regression" 
                else results_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Determine best model
            if problem_type == "Regression":
                best_model_name = max(results.keys(), key=lambda x: results[x]['R² Score'])
                best_metric = "R² Score"
                best_value = results[best_model_name]['R² Score']
            else:
                best_model_name = max(results.keys(), key=lambda x: results[x]['F1 Score'])
                best_metric = "F1 Score"
                best_value = results[best_model_name]['F1 Score']
            
            st.session_state.best_model_name = best_model_name
            st.session_state.best_model = trained_models[best_model_name]['model']
            
            # Best model highlight
            st.markdown("### 🏆 Best Model")
            
            st.markdown(f"""
            <div class="best-model">
                🏆 {best_model_name}<br>
                <span style="font-size: 1.2rem;">{best_metric}: {best_value:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Classification: Show confusion matrices
            if problem_type == "Classification":
                st.markdown("### 🎯 Confusion Matrices")
                
                preprocessing_results = st.session_state.preprocessing_results
                y_test = preprocessing_results['y_test']
                
                cols = st.columns(min(len(trained_models), 2))
                
                for idx, (model_name, model_data) in enumerate(trained_models.items()):
                    with cols[idx % 2]:
                        fig_cm = plot_confusion_matrix(y_test, model_data['predictions'], model_name)
                        st.plotly_chart(fig_cm, use_container_width=True)
            
            # Regression: Actual vs Predicted scatter
            if problem_type == "Regression":
                st.markdown("### 📈 Actual vs Predicted Values")
                
                preprocessing_results = st.session_state.preprocessing_results
                y_test = preprocessing_results['y_test']
                
                model_to_plot = st.selectbox(
                    "Select model to visualize:",
                    options=list(trained_models.keys())
                )
                
                y_pred = trained_models[model_to_plot]['predictions']
                
                fig = go.Figure()
                
                # Scatter plot
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color='#667eea',
                        size=8,
                        opacity=0.6
                    ),
                    hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
                ))
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='#f5576c', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title=f'Actual vs Predicted - {model_to_plot}',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    template='plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ===============================================
    # TAB 4: PREDICTIONS
    # ===============================================
    
    with tabs[3]:
        st.markdown("## 🔮 Make Predictions")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first (Tab 3)")
            return
        
        # Model export section
        st.markdown("### 💾 Export Best Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"🏆 Best Model: **{st.session_state.best_model_name}**")
        
        with col2:
            if st.button("📥 Download Model", use_container_width=True):
                # Prepare model package
                model_package = {
                    'model': st.session_state.best_model,
                    'scaler': st.session_state.preprocessing_results['scaler'],
                    'imputer': st.session_state.preprocessing_results['imputer'],
                    'label_encoders': st.session_state.preprocessing_results['label_encoders'],
                    'y_encoder': st.session_state.preprocessing_results['y_encoder'],
                    'feature_names': st.session_state.feature_columns,
                    'model_name': st.session_state.best_model_name,
                    'problem_type': st.session_state.problem_type
                }
                
                # Serialize
                buffer = io.BytesIO()
                pickle.dump(model_package, buffer)
                buffer.seek(0)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="💾 Download .pkl file",
                    data=buffer,
                    file_name=f"best_model_{timestamp}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Prediction section
        st.markdown("### 🎯 Make New Predictions")
        
        prediction_method = st.radio(
            "Choose prediction method:",
            ["Manual Input", "Upload File"],
            horizontal=True
        )
        
        if prediction_method == "Manual Input":
            st.markdown("#### 📝 Enter Feature Values")
            
            feature_columns = st.session_state.feature_columns
            df = st.session_state.df
            
            input_data = {}
            
            cols = st.columns(min(len(feature_columns), 3))
            
            for idx, col in enumerate(feature_columns):
                with cols[idx % 3]:
                    if df[col].dtype == 'object':
                        unique_vals = df[col].dropna().unique()
                        input_data[col] = st.selectbox(
                            col,
                            options=unique_vals
                        )
                    else:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        mean_val = float(df[col].mean())
                        
                        input_data[col] = st.number_input(
                            col,
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val
                        )
            
            if st.button("🔮 Predict", use_container_width=True, type="primary"):
                try:
                    # Prepare input
                    input_df = pd.DataFrame([input_data])
                    
                    # Apply same preprocessing
                    preprocessing_results = st.session_state.preprocessing_results
                    label_encoders = preprocessing_results['label_encoders']
                    
                    # Encode categorical variables
                    for col in input_df.columns:
                        if col in label_encoders:
                            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    
                    # Impute and scale
                    input_imputed = preprocessing_results['imputer'].transform(input_df)
                    input_scaled = preprocessing_results['scaler'].transform(input_imputed)
                    
                    # Predict
                    prediction = st.session_state.best_model.predict(input_scaled)
                    
                    # Decode if classification
                    if st.session_state.problem_type == "Classification":
                        y_encoder = preprocessing_results['y_encoder']
                        if y_encoder is not None:
                            prediction = y_encoder.inverse_transform(prediction)
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")
                    
                    st.markdown(f"""
                    <div class="best-model">
                        Predicted Value: {prediction[0]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        else:
            st.markdown("#### 📤 Upload New Data")
            
            pred_file = st.file_uploader(
                "Upload file for batch predictions",
                type=['csv', 'xlsx', 'tsv'],
                key="prediction_file"
            )
            
            if pred_file is not None:
                pred_df = load_uploaded_file(pred_file)
                
                if pred_df is not None:
                    st.dataframe(pred_df.head(), use_container_width=True)
                    
                    if st.button("🔮 Generate Predictions", use_container_width=True, type="primary"):
                        try:
                            # Ensure all feature columns are present
                            feature_columns = st.session_state.feature_columns
                            
                            missing_cols = [col for col in feature_columns if col not in pred_df.columns]
                            
                            if missing_cols:
                                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                            else:
                                X_new = pred_df[feature_columns].copy()
                                
                                # Apply preprocessing
                                preprocessing_results = st.session_state.preprocessing_results
                                label_encoders = preprocessing_results['label_encoders']
                                
                                for col in X_new.columns:
                                    if col in label_encoders:
                                        X_new[col] = label_encoders[col].transform(X_new[col].astype(str))
                                
                                X_new_imputed = preprocessing_results['imputer'].transform(X_new)
                                X_new_scaled = preprocessing_results['scaler'].transform(X_new_imputed)
                                
                                predictions = st.session_state.best_model.predict(X_new_scaled)
                                
                                # Decode if needed
                                if st.session_state.problem_type == "Classification":
                                    y_encoder = preprocessing_results['y_encoder']
                                    if y_encoder is not None:
                                        predictions = y_encoder.inverse_transform(predictions)
                                
                                # Add predictions to dataframe
                                pred_df['Prediction'] = predictions
                                
                                st.success("✅ Predictions generated!")
                                
                                st.dataframe(pred_df, use_container_width=True)
                                
                                # Download results
                                csv = pred_df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download Predictions",
                                    data=csv,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")

# ===============================================
# 🚀 RUN APPLICATION
# ===============================================

if __name__ == "__main__":
    main()