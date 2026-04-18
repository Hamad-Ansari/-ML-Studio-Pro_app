# 🧠 ML Studio — Streamlit Machine Learning App

A fully interactive, production-quality ML web application with a modern dark glassmorphism UI.

---

## 🚀 Quick Start

```bash
# 1. Clone / copy files into a folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run ml_studio.py
```

---

## 🗂️ Project Structure

```
ml_studio.py        ← Main app (single-file, fully self-contained)
requirements.txt    ← Python dependencies
README.md           ← This file
```

---

## 🎯 Feature Walkthrough

| Step | What to do |
|------|-----------|
| **1** | Choose **Example Dataset** (iris / titanic / tips) or upload CSV/XLSX/TSV |
| **2** | Click **▶ Run Analysis** in the sidebar |
| **3** | Explore your data in the **📊 Data** tab |
| **4** | Select features, target, problem type & models in the sidebar |
| **5** | Click **🚀 Train Models** — watch the live progress bar |
| **6** | View ranked results & charts in the **📈 Results** tab |
| **7** | Download the best model as `.pkl` |
| **8** | Make predictions (manual sliders or bulk CSV upload) in the **🔮 Predict** tab |

---

## 🤖 Supported Models

### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)

---

## ⚙️ Preprocessing (Automatic)

1. **Categorical Encoding** — `LabelEncoder` per column, stored for inverse-transform
2. **Missing Value Imputation** — `IterativeImputer` (MICE algorithm)
3. **Feature Scaling** — `StandardScaler`

---

## 📊 Evaluation Metrics

| Problem | Metrics |
|---------|---------|
| Regression | R², RMSE, MAE, MSE |
| Classification | Accuracy, F1, Precision, Recall + Confusion Matrix |

---

## 🎨 UI Highlights

- Dark glassmorphism design (blue–purple gradient)
- Syne (display) + DM Sans (body) Google Fonts
- Fully animated — fade/slide-in cards, pulse borders on best model
- Plotly interactive charts (histograms, bar comparisons, scatter, confusion matrix)
- Tab-based layout: Data · Training · Results · Predict
- Metric cards, leaderboard table with highlight, animated training progress bar

---

## 💡 Tips

- **Titanic** → great Classification demo (survived prediction)
- **Tips** → great Regression demo (tip amount prediction)
- **Iris** → great Classification demo (species prediction)
- For your own dataset: ensure the target column is the **last column** (pre-selected by default)
- Large files (>50k rows) may slow SVR/SVC — deselect those models for speed

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `scikit-learn` | ML models & preprocessing |
| `pandas / numpy` | Data manipulation |
| `plotly` | Interactive charts |
| `seaborn` | Example datasets |
| `openpyxl` | Excel file support |
```
https://zcv2ywjvck4imbetwyc4v2.streamlit.app/
```
