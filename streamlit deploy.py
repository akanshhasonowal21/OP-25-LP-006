# ========================= IMPORTS =========================
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="MENTAL WELLNESS AND HEALTH OF THE TECH WORKFORCE",
    page_icon="🧠",
    layout="wide"
)

SMALL_FIGSIZE = (4, 2.5)  # compact charts

# -------------------- FILE PATHS --------------------
CSV_PATH = "data/omsicc.csv"
CLF_PKL = "models/classification_model.pkl"
REG_PKL = "models/regression_model.pkl"
CLU_PKL = "models/clustering_model.pkl"

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data(CSV_PATH)

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model at {path}.\n{e}")
        return None

clf_model = load_model(CLF_PKL)
reg_model = load_model(REG_PKL)
clu_model = load_model(CLU_PKL)

# -------------------- FEATURE LISTS --------------------
CLF_FEATURES = [
    'Age', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'anonymity',
    'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'Sex'
]
REG_FEATURES = [col for col in df.columns if col != "Age"]
CLUSTER_FEATURES = [
    'Age', 'no_employees', 'family_history', 'treatment', 'work_interfere',
    'benefits', 'care_options', 'wellness_program', 'leave'
]

# -------------------- CUSTOM QUESTIONS --------------------
COLUMN_QUESTIONS = {
    'Age': "🎂 What is your age?",
    'self_employed': "💼 Are you self-employed?",
    'family_history': "👨‍👩‍👧‍👦 Do you have a family history of mental illness?",
    'work_interfere': "💼 How often does your mental health interfere with work?",
    'no_employees': "🏢 How many employees work at your company?",
    'remote_work': "🏠 Do you work remotely?",
    'tech_company': "💻 Do you work for a tech company?",
    'benefits': "🏥 Does your employer provide mental health benefits?",
    'care_options': "📋 Do you know the care options available?",
    'wellness_program': "🧘 Is there a wellness program at your workplace?",
    'seek_help': "🆘 Does your employer encourage seeking help?",
    'anonymity': "🕵️‍♂️ Is mental health support kept confidential?",
    'leave': "📝 How easy is it to take medical leave for mental health?",
    'mental_health_consequence': "🧠 Consequences of discussing mental health at work?",
    'phys_health_consequence': "🏋️ Consequences of discussing physical health at work?",
    'coworkers': "👥 Would you talk to your coworkers about mental health?",
    'supervisor': "🧑‍💼 Would you talk to your supervisor about mental health?",
    'mental_health_interview': "💬 Would you discuss mental health in a job interview?",
    'phys_health_interview': "🩺 Would you discuss physical health in a job interview?",
    'mental_vs_physical': "⚖️ Is mental health as important as physical health?",
    'obs_consequence': "👀 Have you observed negative consequences for mental health disclosure?",
    'Sex': "⚧ What is your gender?",
    'treatment': "💊 Have you sought treatment for mental health?"
}

# -------------------- SAFE PREDICT --------------------
def safe_predict(model, X: pd.DataFrame, feature_list: list[str]):
    X_sel = X[feature_list].copy()
    if isinstance(model, Pipeline):
        return model.predict(X_sel)

    for col in X_sel.columns:
        if X_sel[col].dtype == 'object':
            le = LabelEncoder()
            X_sel[col] = le.fit_transform(X_sel[col].astype(str))

    for col in X_sel.columns:
        if X_sel[col].isna().any():
            if np.issubdtype(X_sel[col].dtype, np.number):
                X_sel[col].fillna(0, inplace=True)
            else:
                X_sel[col].fillna(X_sel[col].mode()[0], inplace=True)

    return model.predict(X_sel)

# -------------------- FORM BUILDER --------------------
def build_form_from_columns(df_src: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    inputs = {}
    for col in columns:
        label = COLUMN_QUESTIONS.get(col, col)
        series = df_src[col].dropna()
        if series.dtype.kind in "biufc":
            col_min = float(series.min()) if not series.empty else 0.0
            col_max = float(series.max()) if not series.empty else 100.0
            default = float(series.median()) if not series.empty else 0.0
            inputs[col] = st.number_input(label, value=default, min_value=col_min, max_value=col_max)
        else:
            options = sorted(list(map(str, series.unique()))) if not series.empty else ["N/A"]
            default = options[0]
            inputs[col] = st.selectbox(label, options, index=options.index(default) if default in options else 0)
    return pd.DataFrame([inputs])

# -------------------- NAVIGATION --------------------
st.sidebar.title("🏄‍♂️ Surf the Menu")
section = st.sidebar.radio(
    "📌 Choose Section:",
    ["Overview", "Canvas of the Dataset", "Supervised Model🔍", "Clustering Personas🧘", "About"]
)

# -------------------- OVERVIEW --------------------
if section == "Overview":
    st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>🧠 Mental Wellness & Health of the Tech Workforce 🌱</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>From Awareness to Action: Turning Insights into Change ✨</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Dataset Rows", len(df))
    col2.metric("📐 Features", df.shape[1])
    col3.metric("🤖 Models Loaded", sum(m is not None for m in [clf_model, reg_model, clu_model]))
    st.markdown("### 🎯 Project Goals")
    st.write("""
    - **Identify** those most likely to suffer silently 🕵️‍♂️
    - **Explore** workplace culture impact 💼
    - **Segment** employees into actionable personas 🧩
    """)
    st.markdown("### 👀 Preview of the Dataset")
    st.dataframe(df.head())

# -------------------- EDA WITH INSIGHTS --------------------
elif section == "Canvas of the Dataset":
    st.title("📊 Data Exploration")
    plot_choice = st.radio(
        "Select a visualization:",
        [
            "Number of Employees by Treatment",
            "Age by Treatment",
            "Work Interference (Pie Chart)",
            "Sex by Tech Company",
            "Age by Company Size",
            "Age by Treatment",
            "Correlation Heatmap"
        ]
    )
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    insight_text = ""
    if plot_choice == "Number of Employees by Treatment":
        sns.countplot(data=df, x='no_employees', hue='treatment', palette='pastel', ax=ax)
        group_rates = df.groupby('no_employees')['treatment'].value_counts(normalize=True).unstack().fillna(0)
        top_group = group_rates[1].idxmax() if 1 in group_rates else group_rates.iloc[:, 0].idxmax()
        top_rate = group_rates.loc[top_group].max() * 100
        insight_text = f"💡 Employees in companies with **{top_group}** employees have the highest treatment-seeking rate at **{top_rate:.1f}%**."
    elif plot_choice == "Age by Treatment":
        sns.histplot(data=df, x='Age', hue='treatment', multiple='dodge', palette='Set2', ax=ax)
        avg_age_yes = df.loc[df['treatment'] == 1, 'Age'].mean()
        avg_age_no = df.loc[df['treatment'] == 0, 'Age'].mean()
        insight_text = f"💡 Average age of those seeking treatment is **{avg_age_yes:.1f}**, compared to **{avg_age_no:.1f}**."
    elif plot_choice == "Work Interference (Pie Chart)":
        counts = df['work_interfere'].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        top_category = counts.idxmax()
        top_percent = counts.max() / counts.sum() * 100
        insight_text = f"💡 Most common work interference level: **{top_category}** ({top_percent:.1f}% of respondents)."
    elif plot_choice == "Sex by Tech Company":
        sns.countplot(data=df, x='Sex', hue='tech_company', palette='coolwarm', ax=ax)
        tech_male_ratio = df[df['tech_company'] == 1]['Sex'].value_counts(normalize=True).max() * 100
        insight_text = f"💡 In tech companies, the largest gender group makes up **{tech_male_ratio:.1f}%** of employees."
    elif plot_choice == "Age by Company Size":
        sns.boxplot(data=df, x='no_employees', y='Age', palette='viridis', ax=ax)
        youngest_group = df.groupby('no_employees')['Age'].median().idxmin()
        insight_text = f"💡 Youngest median age is in companies with **{youngest_group}** employees."
    elif plot_choice == "Age by Treatment":
        sns.violinplot(data=df, x='treatment', y='Age', palette='muted', ax=ax)
        youngest_treatment = df.groupby('treatment')['Age'].median().idxmin()
        insight_text = f"💡 {'Those not seeking treatment' if youngest_treatment == 0 else 'Those seeking treatment'} tend to be younger."
    elif plot_choice == "Correlation Heatmap":
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        corr_unstacked = corr.unstack().dropna()
        strongest_pos = corr_unstacked[corr_unstacked < 1].idxmax()
        strongest_neg = corr_unstacked.idxmin()
        insight_text = f"💡 Strongest positive correlation: **{strongest_pos[0]}** & **{strongest_pos[1]}**; strongest negative correlation: **{strongest_neg[0]}** & **{strongest_neg[1]}**."
    st.pyplot(fig, use_container_width=False)
    if insight_text:
        st.info(insight_text)

# -------------------- SUPERVISED MODEL --------------------
elif section == "Supervised Model🔍":
    sub_section = st.sidebar.radio("Select task:", ["Classification", "Regression"])
    if sub_section == "Classification":
        st.title("🤖 Classification: Treatment Prediction")
        if clf_model is not None:
            st.subheader("🔍 Know Your Prediction")
            X_row = build_form_from_columns(df, CLF_FEATURES)
            pred = safe_predict(clf_model, X_row, CLF_FEATURES)[0]
            label = "seek" if str(pred) in ["1", "Yes", "True"] else "not seek"
            st.success(f"Prediction: Employee will **{label}** treatment.")
        else:
            st.error("Classification model not loaded.")
    elif sub_section == "Regression":
        st.title("📈 Regression: Age Prediction")
        if reg_model is not None:
            st.subheader("🔍 Know Your Prediction")
            X_row = build_form_from_columns(df, REG_FEATURES)
            y_pred = safe_predict(reg_model, X_row, REG_FEATURES)[0]
            st.success(f"Predicted Age: **{float(y_pred):.1f}** years")
        else:
            st.error("Regression model not loaded.")

# -------------------- CLUSTERING --------------------
elif section == "Clustering Personas🧘":
    st.title("🌀 Clustering: Mental Health Personas")
    if clu_model is not None:
        st.subheader("🔍 Know Your Prediction")
        X_clu = build_form_from_columns(df, CLUSTER_FEATURES)
        cluster = safe_predict(clu_model, X_clu, CLUSTER_FEATURES)[0]

        cluster_names = {
            0: "Silent Sufferers",
            1: "Open Advocates",
            2: "Under-Supported Professionals"
        }
        cluster_descriptions = {
            "Silent Sufferers": "High stress levels, low openness to seeking help, often without adequate workplace support.",
            "Open Advocates": "Comfortable discussing mental health and have strong workplace policies and benefits in place.",
            "Under-Supported Professionals": "Willing to engage about mental health but lack sufficient resources or support at work."
        }

        persona_name = cluster_names.get(cluster, f"Cluster {cluster}")
        st.success(f"Persona: **{persona_name}**")
        if persona_name in cluster_descriptions:
            st.info(cluster_descriptions[persona_name])
    else:
        st.error("Clustering model not loaded.")

# ========================= DOWNLOAD ARTIFACTS =========================
elif section == "Download Artifacts":
    def section_title(title: str, subtitle: str = ""):
        st.subheader(title)
        if subtitle:
            st.write(subtitle)
    section_title("Artifacts", "Export and share")

    if "clf_pipeline" in st.session_state:
        buf = io.BytesIO()
        joblib.dump(st.session_state["clf_pipeline"], buf)
        st.download_button("Download Classification Pipeline (.joblib)", data=buf.getvalue(), file_name="classification_pipeline.joblib")
    else:
        st.info("Train a classification model to enable pipeline download.")

    if "reg_pipeline" in st.session_state:
        buf2 = io.BytesIO()
        joblib.dump(st.session_state["reg_pipeline"], buf2)
        st.download_button("Download Regression Pipeline (.joblib)", data=buf2.getvalue(), file_name="regression_pipeline.joblib")
    else:
        st.info("Train a regression model to enable pipeline download.")

    if "cluster_assignments" in st.session_state:
        csv = st.session_state["cluster_assignments"].to_csv(index=False).encode("utf-8")
        st.download_button("Download Cluster Assignments (CSV)", data=csv, file_name="cluster_assignments.csv", mime="text/csv")
    else:
        st.info("Run clustering to export assignments.")

# ========================= ABOUT =========================
elif section == "About":
    import streamlit as st
    def section_title(title: str, subtitle: str = ""):
        st.subheader(title)
        if subtitle:
            st.write(subtitle)

        section_title("About this App")
        st.write("""
          **Project Style & Syntax**
                  - Focused on **who avoids treatment**, **policy impacts**, and **actionable personas**.
                  - Clean handling of **age/sex anomalies**, with robust imputations.
                  - Visuals via **Plotly** for interactive analysis; Matplotlib/Seaborn optional.
                  - Models wrapped in **sklearn Pipelines** to keep preprocessing & inference consistent.
    
                   **Personas (plain-English)**
                   - *Silent Sufferers*: Low openness/support signals, prior issues highest risk of not seeking help.
                   - *Open Advocates*: Comfortable discussing mental health, supported by managers & benefits.
                   - *Under-Supported Professionals*: Will engage if support improves; target with benefits & manager training.
                   - *Supported & Aware*: Doing fine; maintain best practices and continuous feedback loops.
    
                  **Next Steps**
                   - Integrate HR policy simulations (what-if benefits, leave flexibility).
                   - Add fairness & bias checks across demographics.
                   - Connect to anonymized engagement data for drift monitoring.

                    WITH LOVE OPEN LEARN'S PIONEERS✨
                """)

