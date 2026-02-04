import streamlit as st
import pandas as pd
import json
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Survey Pipeline Studio")

st.title("🧠 Survey Analytics Studio")

uploaded = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded:

    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded)

    st.success("Dataset loaded")
    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])

    # -------------------- PREVIEW --------------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(200), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    all_cols = df.columns.tolist()

    # -------------------- PROFILER --------------------
    st.subheader("📊 Dataset Profiler")

    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing %", round(df.isna().mean().mean()*100,2))

    st.write("Top Null Columns")
    st.dataframe(df.isna().mean().sort_values(ascending=False).head(10))

    # -------------------- CORRELATION --------------------
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        fig,ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -------------------- CORE SETTINGS --------------------
    st.subheader("⚙️ Core Settings")

    id_col = st.selectbox("ID Column (optional)", ["None"]+all_cols)
    weight_col = st.selectbox("Weight Column (optional)", ["None"]+numeric_cols)

    target = st.selectbox("Target Variable", all_cols)
    predictors = st.multiselect("Predictors", numeric_cols+cat_cols)

    # Auto detect model type
    if df[target].nunique() <= 15:
        model_type = "classification"
    else:
        model_type = "prediction"

    st.info(f"Auto-detected model type: {model_type}")

    # Target distribution
    st.subheader("🎯 Target Distribution")
    st.bar_chart(df[target].value_counts().head(20))

    # -------------------- CLEANING --------------------
    st.subheader("🧹 Cleaning")

    num_impute = st.multiselect("Numeric Imputation", numeric_cols, default=numeric_cols[:1])
    cat_impute = st.multiselect("Categorical Imputation", cat_cols)

    col_null = st.slider("Drop Columns > % Null", 0,100,100)
    row_null = st.slider("Drop Rows > % Null", 0,100,80)

    run_pca = st.checkbox("Enable PCA", True)
    drop_outliers = st.checkbox("Enable Outlier Detection", True)
    outlier_pct = st.slider("Outlier %", 0.0,10.0,1.0)

    # -------------------- GENERATE --------------------
    if st.button("🚀 Generate Config + Run Pipeline"):

        required = predictors + [target]
        if id_col!="None": required.append(id_col)
        if weight_col!="None": required.append(weight_col)

        column_types = {}
        for c in num_impute: column_types[c]="integer"
        for c in cat_impute: column_types[c]="category"

        config = {
            "required_fields": list(set(required)),
            "column_types": column_types,
            "weight_column": None if weight_col=="None" else weight_col,

            "custom_module_settings":{
                "performance_settings":{
                    "run_pca_analysis":run_pca,
                    "run_normality_plots":False
                },

                "cleaner":{
                    "numeric_cols_for_imputation":num_impute,
                    "categorical_cols_for_imputation":cat_impute,
                    "row_null_threshold":row_null,
                    "column_null_threshold":col_null,
                    "rules_based_imputation":[],
                    "imputation_strategy":"mice",
                    "mice_max_iter":5,
                    "mice_n_estimators":10,
                    "drop_outliers":drop_outliers,
                    "outlier_detection_strategy":"isolation_forest",
                    "outlier_removal_percent":outlier_pct
                },

                "weighting":{
                    "analysis_columns":predictors,
                    "raking_config":{"apply_raking":False}
                },

                "analysis_config":{
                    "run_analysis":True,
                    "test_size":0.3,
                    "random_state":42,
                    "model_type":model_type,
                    "target_variable":target,
                    "predictor_variables":predictors,
                    "models_to_run":"find_best"
                }
            }
        }

        with open("pipeline_config.json","w") as f:
            json.dump(config,f,indent=4)

        with open("survey_data.csv","wb") as f:
            f.write(uploaded.getbuffer())

        st.success("Config generated")

        with st.spinner("Running pipeline..."):
            proc = subprocess.run(["python","survey_pipeline.py"],capture_output=True,text=True)

        st.text(proc.stdout)
        st.text(proc.stderr)

        st.success("Pipeline completed")
