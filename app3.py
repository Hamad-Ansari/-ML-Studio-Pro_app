# app.py
# ==========================================================
# 🚀 Hammad AI ML Studio - Production Grade Streamlit App
# Run: streamlit run app.py
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import io
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# SKLEARN
# ==========================================================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Hammad AI ML Studio",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}
.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#2563eb);
    color:white;
    border:none;
    border-radius:12px;
    padding:0.65rem 1rem;
    font-weight:700;
}
.stDownloadButton>button {
    background: linear-gradient(90deg,#10b981,#059669);
    color:white;
    border:none;
    border-radius:12px;
}
section[data-testid="stSidebar"] {
    background:#0f172a;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.title("🤖 Hammad AI ML Studio")
st.markdown("""
### Welcome Hammad 👋  
Upload data, train ML models, compare results, predict outcomes, and download your model instantly.
""")

# ==========================================================
# CACHE DATA LOADERS
# ==========================================================
@st.cache_data
def load_example_dataset(name):
    return sns.load_dataset(name)

@st.cache_data
def load_uploaded_file(file):
    ext = file.name.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(file)

    elif ext == "xlsx":
        return pd.read_excel(file)

    elif ext == "tsv":
        return pd.read_csv(file, sep="\t")

    else:
        return pd.read_csv(file)

# ==========================================================
# SIDEBAR DATA SOURCE
# ==========================================================
st.sidebar.header("📁 Dataset Source")

source = st.sidebar.radio(
    "Choose Source",
    ["Upload Dataset", "Example Dataset"]
)

df = None

if source == "Upload Dataset":

    uploaded = st.sidebar.file_uploader(
        "Upload CSV / XLSX / TSV",
        type=["csv", "xlsx", "tsv"]
    )

    if uploaded:
        df = load_uploaded_file(uploaded)

else:

    dataset_name = st.sidebar.selectbox(
        "Choose Dataset",
        ["titanic", "tips", "iris"]
    )

    df = load_example_dataset(dataset_name)

# ==========================================================
# IF DATA EXISTS
# ==========================================================
if df is not None:

    st.success("✅ Dataset Loaded Successfully")

    # ======================================================
    # DATA PREVIEW
    # ======================================================
    tab1, tab2, tab3 = st.tabs(
        ["📄 Head", "📊 Info", "📈 Describe"]
    )

    with tab1:
        st.dataframe(df.head(), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])

        with c2:
            st.write("📌 Column Names")
            st.write(list(df.columns))

    with tab3:
        st.dataframe(df.describe(include="all"), use_container_width=True)

    st.markdown("---")

    # ======================================================
    # CONFIG
    # ======================================================
    st.subheader("⚙️ Configuration")

    c1, c2 = st.columns(2)

    with c1:
        problem_type = st.selectbox(
            "Select Problem Type",
            ["Classification", "Regression"]
        )

    with c2:
        target_col = st.selectbox(
            "Select Target Column",
            df.columns
        )

    feature_cols = st.multiselect(
        "Select Feature Columns",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col][:3]
    )

    test_size = st.slider(
        "Train/Test Split %",
        10, 50, 20
    )

    # ======================================================
    # MODEL OPTIONS
    # ======================================================
    st.sidebar.header("🧠 Model")

    if problem_type == "Regression":

        model_name = st.sidebar.selectbox(
            "Choose Model",
            [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine"
            ]
        )

    else:

        model_name = st.sidebar.selectbox(
            "Choose Model",
            [
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine"
            ]
        )

    # ======================================================
    # RUN BUTTON
    # ======================================================
    run = st.button("🚀 Run Analysis & Train")

    # ======================================================
    # TRAINING START
    # ======================================================
    if run:

        if len(feature_cols) == 0:
            st.warning("Please choose feature columns.")
            st.stop()

        data = df[feature_cols + [target_col]].copy()

        X = data[feature_cols].copy()
        y = data[target_col].copy()

        # ==================================================
        # FEATURE ENCODING
        # ==================================================
        encoders = {}

        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):

                le = LabelEncoder()
                X[col] = X[col].astype(str)
                X[col] = le.fit_transform(X[col])

                encoders[col] = le

        # ==================================================
        # TARGET ENCODING
        # ==================================================
        target_encoder = None

        if problem_type == "Classification":
            if y.dtype == "object" or str(y.dtype).startswith("category"):

                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))

        # ==================================================
        # IMPUTATION
        # ==================================================
        imputer = IterativeImputer(random_state=42)

        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )

        # ==================================================
        # SCALING
        # ==================================================
        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        # ==================================================
        # SPLIT
        # ==================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size/100,
            random_state=42
        )

        # ==================================================
        # MODELS
        # ==================================================
        if problem_type == "Regression":

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Support Vector Machine": SVR()
            }

        else:

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC(probability=True)
            }

        model = models[model_name]

        # ==================================================
        # TRAIN
        # ==================================================
        with st.spinner("Training Model..."):

            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        st.success("✅ Training Completed")

        # ==================================================
        # EVALUATION
        # ==================================================
        st.subheader("📊 Evaluation")

        # --------------------------------------------------
        # REGRESSION
        # --------------------------------------------------
        if problem_type == "Regression":

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            result = pd.DataFrame({
                "Metric": ["MSE", "RMSE", "MAE", "R2 Score"],
                "Value": [mse, rmse, mae, r2]
            })

            st.dataframe(result, use_container_width=True)

        # --------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------
        else:

            acc = accuracy_score(y_test, preds)
            pre = precision_score(
                y_test, preds,
                average="weighted",
                zero_division=0
            )
            rec = recall_score(
                y_test, preds,
                average="weighted",
                zero_division=0
            )
            f1 = f1_score(
                y_test, preds,
                average="weighted",
                zero_division=0
            )

            result = pd.DataFrame({
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ],
                "Value": [acc, pre, rec, f1]
            })

            st.dataframe(result, use_container_width=True)

            cm = confusion_matrix(y_test, preds)

            st.write("### Confusion Matrix")
            st.dataframe(cm)

        st.success(f"🏆 Best Model: {model_name}")

        # ==================================================
        # DOWNLOAD MODEL
        # ==================================================
        st.subheader("⬇️ Download Model")

        model_bytes = pickle.dumps(model)

        st.download_button(
            label="Download Pickle Model",
            data=model_bytes,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

        # ==================================================
        # PREDICTION SECTION
        # ==================================================
        st.subheader("🔮 Make Prediction")

        input_values = {}

        for col in feature_cols:
            input_values[col] = st.number_input(
                f"Enter {col}",
                value=0.0
            )

        if st.button("Predict"):

            pred_df = pd.DataFrame([input_values])

            # Encode if needed
            for col in pred_df.columns:
                if col in encoders:
                    pred_df[col] = encoders[col].transform(
                        pred_df[col].astype(str)
                    )

            pred_df = pd.DataFrame(
                imputer.transform(pred_df),
                columns=pred_df.columns
            )

            pred_df = scaler.transform(pred_df)

            pred = model.predict(pred_df)[0]

            if target_encoder:
                pred = target_encoder.inverse_transform(
                    [int(pred)]
                )[0]

            st.success(f"🎯 Prediction: {pred}")

else:
    st.info("Upload a dataset or choose example dataset from sidebar.")