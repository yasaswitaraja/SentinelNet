import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelNet IDS",
    page_icon="🔐",
    layout="wide"
)

# ──────────────────────────────────────────────
#  Custom CSS  (dark cyber theme)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #c9d1d9; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        padding: 10px;
    }
    div.stButton > button:hover { background-color: #2ea043; }

    /* Headings */
    h1, h2, h3 { color: #58a6ff; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

    /* Success / Info / Warning boxes */
    div[data-testid="stAlert"] { border-radius: 8px; }

    /* Section divider */
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("⚡ SentinelNet IDS")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

st.sidebar.markdown("### 🤖 Choose ML Model")
model_choice = st.sidebar.radio(
    "",
    ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### ⚙️ Settings")
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
show_report = st.sidebar.checkbox("Show Full Classification Report", value=False)
show_feature_imp = st.sidebar.checkbox("Show Feature Importance", value=True)

st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Run Intrusion Detection")

st.sidebar.markdown("""
---
**ℹ️ Supported Datasets**
- NSL-KDD
- CICIDS2017
- Any labeled CSV
""")

# ──────────────────────────────────────────────
#  Main Page Header
# ──────────────────────────────────────────────
st.title("🔐 SentinelNet — Network Intrusion Detection System")
st.markdown(
    "Upload a labeled network traffic CSV, select a model, and detect intrusions. "
    "The system auto-handles preprocessing, training, evaluation, and visualization."
)
st.markdown("---")

# ──────────────────────────────────────────────
#  Helper: Attack type color map for NSL-KDD
# ──────────────────────────────────────────────
ATTACK_COLORS = {
    "normal": "#2ea043",
    "dos":    "#f85149",
    "probe":  "#d29922",
    "r2l":    "#a371f7",
    "u2r":    "#ff7b72",
}

def get_label_color(label):
    label_lower = str(label).lower()
    for key, color in ATTACK_COLORS.items():
        if key in label_lower:
            return color
    return "#58a6ff"

# ──────────────────────────────────────────────
#  Main Logic
# ──────────────────────────────────────────────
if uploaded_file is None:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1️⃣**\nUpload a CSV network dataset from the sidebar.")
    with col2:
        st.info("**Step 2️⃣**\nChoose an ML model and configure settings.")
    with col3:
        st.info("**Step 3️⃣**\nClick **Run Intrusion Detection** to train & evaluate.")

    st.markdown("---")
    st.markdown("### 📌 Supported Attack Categories")
    cols = st.columns(5)
    labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    descs  = ["Legitimate traffic", "Denial of Service", "Reconnaissance", "Remote to Local", "User to Root"]
    for col, lbl, desc in zip(cols, labels, descs):
        color = get_label_color(lbl)
        col.markdown(f"""
        <div style="background:{color}22; border:1px solid {color};
                    border-radius:8px; padding:12px; text-align:center;">
            <b style="color:{color}">{lbl}</b><br>
            <small>{desc}</small>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Dataset loaded ──
df = pd.read_csv(uploaded_file)
st.success(f"✅ Dataset loaded — **{df.shape[0]:,} rows × {df.shape[1]} columns**")

# Dataset preview
with st.expander("🔍 Preview Dataset", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

# Class distribution
target_col = df.columns[-1]
st.markdown(f"**🎯 Target column detected:** `{target_col}`")

col_dist, col_bar = st.columns([1, 2])
with col_dist:
    st.markdown("**Class Distribution**")
    dist = df[target_col].value_counts().reset_index()
    dist.columns = ["Class", "Count"]
    dist["% Share"] = (dist["Count"] / dist["Count"].sum() * 100).round(2)
    st.dataframe(dist, use_container_width=True)

with col_bar:
    fig_dist, ax_dist = plt.subplots(figsize=(6, 3))
    fig_dist.patch.set_facecolor("#0d1117")
    ax_dist.set_facecolor("#161b22")
    labels_list = dist["Class"].astype(str).tolist()
    counts_list  = dist["Count"].tolist()
    bar_colors   = [get_label_color(l) for l in labels_list]
    bars = ax_dist.barh(labels_list, counts_list, color=bar_colors, edgecolor="#30363d")
    ax_dist.set_xlabel("Count", color="#8b949e")
    ax_dist.tick_params(colors="#c9d1d9")
    ax_dist.spines[:].set_color("#30363d")
    for spine in ax_dist.spines.values():
        spine.set_edgecolor("#30363d")
    ax_dist.set_title("Traffic Class Distribution", color="#58a6ff")
    plt.tight_layout()
    st.pyplot(fig_dist)

st.markdown("---")

# ── Training ──
if not train_button:
    st.info("👈 Configure settings in the sidebar and click **Run Intrusion Detection** to start.")
    st.stop()

with st.spinner("🔄 Preprocessing data and training model..."):

    # ── Preprocessing ──
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # Encode categorical target
    original_labels = None
    if y.dtype == "object":
        le_target = LabelEncoder()
        original_labels = list(y.unique())
        y = le_target.fit_transform(y)
        label_names = le_target.classes_
    elif pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        # Continuous target → binary
        y = (y > y.median()).astype(int)
        label_names = ["Normal (≤median)", "Attack (>median)"]
        st.info("ℹ️ Continuous target detected — converted to binary using median threshold.")
    else:
        label_names = [str(c) for c in sorted(y.unique())]

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Fill missing values
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model ──
    model_map = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
    }
    model = model_map[model_choice]
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # ── Metrics ──
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# ── Results ──
st.markdown(f"## 📊 Results — `{model_choice}`")

m1, m2, m3, m4 = st.columns(4)
m1.metric("🎯 Accuracy",  f"{acc*100:.2f}%")
m2.metric("🔍 Precision", f"{prec*100:.2f}%")
m3.metric("📡 Recall",    f"{rec*100:.2f}%")
m4.metric("⚖️ F1 Score",  f"{f1*100:.2f}%")

st.markdown("---")

# ── Confusion Matrix + Feature Importance ──
col_cm, col_fi = st.columns(2)

with col_cm:
    st.markdown("### 🔢 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    fig_cm.patch.set_facecolor("#0d1117")
    ax_cm.set_facecolor("#161b22")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix", color="#58a6ff")
    ax_cm.tick_params(colors="#c9d1d9", labelsize=8)
    ax_cm.xaxis.label.set_color("#8b949e")
    ax_cm.yaxis.label.set_color("#8b949e")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig_cm)

with col_fi:
    if show_feature_imp and hasattr(model, "feature_importances_"):
        st.markdown("### 📈 Top 10 Feature Importances")
        importances = model.feature_importances_
        feat_names  = X.columns.tolist()
        top_idx     = np.argsort(importances)[-10:][::-1]
        top_feats   = [feat_names[i] for i in top_idx]
        top_vals    = importances[top_idx]

        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        fig_fi.patch.set_facecolor("#0d1117")
        ax_fi.set_facecolor("#161b22")
        bars = ax_fi.barh(top_feats[::-1], top_vals[::-1], color="#58a6ff", edgecolor="#30363d")
        ax_fi.set_xlabel("Importance Score", color="#8b949e")
        ax_fi.tick_params(colors="#c9d1d9", labelsize=8)
        ax_fi.spines[:].set_color("#30363d")
        ax_fi.set_title("Feature Importances", color="#58a6ff")
        plt.tight_layout()
        st.pyplot(fig_fi)
    elif show_feature_imp and hasattr(model, "coef_"):
        st.markdown("### 📈 Top 10 Feature Coefficients (Logistic Regression)")
        coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        feat_names = X.columns.tolist()
        top_idx    = np.argsort(coef)[-10:][::-1]
        top_feats  = [feat_names[i] for i in top_idx]
        top_vals   = coef[top_idx]

        fig_lr, ax_lr = plt.subplots(figsize=(5, 4))
        fig_lr.patch.set_facecolor("#0d1117")
        ax_lr.set_facecolor("#161b22")
        ax_lr.barh(top_feats[::-1], top_vals[::-1], color="#a371f7", edgecolor="#30363d")
        ax_lr.set_xlabel("|Coefficient|", color="#8b949e")
        ax_lr.tick_params(colors="#c9d1d9", labelsize=8)
        ax_lr.spines[:].set_color("#30363d")
        ax_lr.set_title("Feature Coefficients", color="#58a6ff")
        plt.tight_layout()
        st.pyplot(fig_lr)
    else:
        st.info("Enable **Show Feature Importance** in the sidebar to view this chart.")

# ── Classification Report ──
if show_report:
    st.markdown("---")
    st.markdown("### 📋 Full Classification Report")
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

# ── Sample Predictions ──
st.markdown("---")
st.markdown("### 🔎 Sample Predictions (First 20 test rows)")
sample_df = X_test.copy().reset_index(drop=True).head(20)
sample_df["Actual"]    = [label_names[i] if i < len(label_names) else str(i) for i in y_test[:20]]
sample_df["Predicted"] = [label_names[i] if i < len(label_names) else str(i) for i in y_pred[:20]]
sample_df["✅ Match"]  = sample_df["Actual"] == sample_df["Predicted"]

def highlight_match(row):
    color = "#1f3a1f" if row["✅ Match"] else "#3a1f1f"
    return [f"background-color: {color}"] * len(row)

st.dataframe(
    sample_df[["Actual", "Predicted", "✅ Match"]].style.apply(highlight_match, axis=1),
    use_container_width=True
)

# ── Footer ──
st.markdown("---")
st.markdown(
    "<center><small>SentinelNet IDS · Built with Streamlit + Scikit-learn · "
    "Datasets: NSL-KDD / CICIDS2017</small></center>",
    unsafe_allow_html=True
)