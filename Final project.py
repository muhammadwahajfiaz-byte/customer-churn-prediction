import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Feature Selection
from sklearn.feature_selection import f_classif

# Machine Learning
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Streamlit Configuration & Custom Tri-Color Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Churn Analytics Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Background & Main Structure */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8e3f5 50%, #fefce8 100%);
        color: #1e293b !important;
    }
    
    /* Sidebar Styling - Deep Blue & Lilac */
    [data-testid="stSidebar"] {
        background-color: #1e3a8a !important;
        border-right: 3px solid #c8a2c8;
    }
    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Metric Cards - Blue Base with Lilac Border & Lemon Highlights */
    .metric-card {
        background: #ffffff;
        border: 2px solid #c8a2c8;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.08);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #2563eb;
    }
    .metric-title {
        color: #1e3a8a;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #2563eb;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 8px;
    }
    
    /* Lemon Yellow Action Buttons with Blue Text */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #facc15 0%, #fef08a 100%);
        color: #1e3a8a !important;
        font-weight: 800 !important;
        border: 2px solid #eab308;
        border-radius: 10px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(250, 204, 21, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #eab308 0%, #facc15 100%);
        transform: scale(1.01);
    }
    
    /* Banner Gradient - Blue to Lilac with Lemon Accents */
    .banner-container {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #c8a2c8 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        border-left: 8px solid #facc15;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);
    }
    
    label, .stSelectbox label, .stSlider label {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="banner-container">
    <h1 style="color: #ffffff; margin: 0; font-size: 2.3rem;">🔮 Customer Churn & Retention Analytics</h1>
    <p style="color: #fef08a; margin-top: 8px; font-size: 1.05rem;">
        Predict customer attrition, evaluate drivers rapidly, and automate targeted retention offers.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Session State Initialization
if "df" not in st.session_state:
    st.session_state.df = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None


# ---------------------------------------------------------
# Cached Feature Scoring Engine
# ---------------------------------------------------------
@st.cache_data
def get_fast_feature_scores(df_raw, target_col, drop_cols):
    X_raw = df_raw.drop(columns=[target_col] + drop_cols, errors="ignore")
    y_raw = df_raw[target_col]

    y = (
        y_raw.map({"Yes": 1, "No": 0, "True": 1, "False": 0, 1: 1, 0: 0})
        .fillna(0)
        .astype(int)
    )

    X_numeric = X_raw.copy()
    for col in X_numeric.select_dtypes(include=["object", "category"]).columns:
        X_numeric[col] = pd.to_numeric(
            X_numeric[col], errors="coerce"
        ).fillna(X_numeric[col].astype("category").cat.codes)

    X_numeric = X_numeric.fillna(0)
    scores, _ = f_classif(X_numeric, y)

    df_imp = pd.DataFrame({
        "Column Name": X_raw.columns,
        "Statistical Score": [0 if np.isnan(s) else s for s in scores],
    }).sort_values(by="Statistical Score", ascending=False).reset_index(drop=True)

    return df_imp


# Sidebar Navigation
st.sidebar.title("📌 Navigation Menu")
page = st.sidebar.radio(
    "Go to Page",
    [
        "1. Data Ingestion Hub",
        "2. Automated Pipeline",
        "3. Feature Selection Engine",
        "4. Model Training Studio",
        "5. Performance Evaluation",
        "6. AI Retention Engine",
    ],
)

# ---------------------------------------------------------
# PAGE 1: Data Ingestion Hub
# ---------------------------------------------------------
if page == "1. Data Ingestion Hub":
    st.header("📂 1. Data Ingestion & Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

    with col2:
        if st.session_state.df is not None:
            df = st.session_state.df

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    '<div class="metric-card"><div'
                    ' class="metric-title">Total Records</div><div'
                    f' class="metric-value">{df.shape[0]:,}</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    '<div class="metric-card"><div'
                    ' class="metric-title">Total Features</div><div'
                    f' class="metric-value">{df.shape[1]}</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                missing = df.isnull().sum().sum()
                st.markdown(
                    '<div class="metric-card"><div'
                    ' class="metric-title">Missing Values</div><div'
                    ' class="metric-value" style="color:'
                    f' {"#ef4444" if missing > 0 else "#16a34a"};">{missing}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("### Dataset Preview")
            st.dataframe(df.head(6), use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: Data Processing Pipeline
# ---------------------------------------------------------
elif page == "2. Automated Pipeline":
    st.header("⚙️ 2. Preprocessing & Pipeline Construction")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload your dataset on Page 1 first.")
    else:
        df = st.session_state.df.copy()

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(
                "Select Target Variable (Churn)",
                df.columns,
                index=(
                    df.columns.get_loc("Churn")
                    if "Churn" in df.columns
                    else 0
                ),
            )
        with col2:
            drop_cols = st.multiselect(
                "Exclude Unique Key Columns",
                df.columns.tolist(),
                default=(
                    ["customerID"]
                    if "customerID" in df.columns
                    else []
                ),
            )

        if st.button("🚀 Process & Build Pipeline"):
            X = df.drop(columns=[target_col] + drop_cols)
            y = df[target_col]

            if y.dtype == "object":
                y = y.map({"Yes": 1, "No": 0, "True": 1, "False": 0}).fillna(0)

            num_cols = X.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            cat_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            num_pipeline = Pipeline([("scaler", StandardScaler())])
            cat_pipeline = Pipeline([
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ])

            X_processed = preprocessor.fit_transform(X)

            st.session_state.X = X_processed
            st.session_state.y = y
            st.session_state.preprocessor = preprocessor

            st.success("✅ Preprocessing Complete!")

            if target_col in df.columns:
                fig = px.pie(
                    df,
                    names=target_col,
                    title="Class Distribution",
                    hole=0.4,
                    color_discrete_sequence=["#1e3a8a", "#c8a2c8"],
                )
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: Feature Selection Engine
# ---------------------------------------------------------
elif page == "3. Feature Selection Engine":
    st.header("🎯 3. Fast Filter-Based Feature Selection")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload your dataset on Page 1 first.")
    else:
        df = st.session_state.df.copy()

        target_col = "Churn" if "Churn" in df.columns else df.columns[-1]
        drop_cols = (
            ["customerID"]
            if "customerID" in df.columns
            else [df.columns[0]]
        )

        X_raw = df.drop(columns=[target_col] + drop_cols, errors="ignore")

        top_k = st.slider(
            "Select Top Columns to Retain",
            min_value=2,
            max_value=len(X_raw.columns),
            value=min(10, len(X_raw.columns)),
        )

        df_imp = get_fast_feature_scores(df, target_col, drop_cols)

        st.session_state.selected_raw_columns = df_imp.head(top_k)[
            "Column Name"
        ].tolist()

        st.success(
            f"⚡ Calculated instantly! Kept top **{top_k}** columns for"
            " training."
        )

        fig = px.bar(
            df_imp,
            x="Statistical Score",
            y="Column Name",
            orientation="h",
            title="Filter Method Column Scores",
            color="Statistical Score",
            color_continuous_scale=["#fef08a", "#c8a2c8", "#1e3a8a"],
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_imp, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: High Performance Model Training (Balanced Class Weights)
# ---------------------------------------------------------
elif page == "4. Model Training Studio":
    st.header("🤖 4. Model Training Studio")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload your dataset on Page 1 first.")
    else:
        run_cv = st.checkbox(
            "Enable 3-Fold Cross Validation (Optional)", value=False
        )

        if st.button("🔥 Train Machine Learning Models"):
            with st.spinner("Training models with class balancing..."):
                df = st.session_state.df.copy()

                target_col = "Churn" if "Churn" in df.columns else df.columns[-1]
                drop_cols = (
                    ["customerID"]
                    if "customerID" in df.columns
                    else [df.columns[0]]
                )

                if "selected_raw_columns" in st.session_state:
                    selected_cols = st.session_state.selected_raw_columns
                    X_raw = df[selected_cols]
                else:
                    X_raw = df.drop(columns=[target_col] + drop_cols, errors="ignore")

                y_raw = df[target_col]
                y_numeric = (
                    y_raw.map(
                        {"Yes": 1, "No": 0, "True": 1, "False": 0, 1: 1, 0: 0}
                    )
                    .fillna(0)
                    .astype(int)
                    .values
                )

                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw,
                    y_numeric,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_numeric,
                )

                num_cols = X_raw.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()
                cat_cols = X_raw.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                preprocessor = ColumnTransformer(transformers=[
                    ("num", StandardScaler(), num_cols),
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                        cat_cols,
                    ),
                ])

                X_train_processed = preprocessor.fit_transform(X_train_raw,y_train)
                X_test_processed = preprocessor.transform(X_test_raw)

                # Class Weights applied to boost Recall and F1-Scores (~80% target)
                models = {
                    "Logistic Regression": LogisticRegression(
                        max_iter=200, class_weight="balanced", random_state=42
                    ),
                    "Decision Tree": DecisionTreeClassifier(
                        max_depth=5, class_weight="balanced", random_state=42
                    ),
                    "Random Forest": RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                    "XGBoost": XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        scale_pos_weight=2.7,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                }

                results = []
                trained_models = {}

                progress_bar = st.progress(0)
                for idx, (name, model) in enumerate(models.items()):
                    model.fit(X_train_processed, y_train)

                    preds = model.predict(X_test_processed)
                    probs = (
                        model.predict_proba(X_test_processed)[:, 1]
                        if hasattr(model, "predict_proba")
                        else preds
                    )

                    cv_score = "N/A"
                    if run_cv:
                        cv_score = round(
                            cross_val_score(
                                model,
                                X_train_processed,
                                y_train,
                                cv=3,
                                scoring="f1",
                            ).mean(),
                            4,
                        )

                    results.append({
                        "Model": name,
                        "Accuracy": round(accuracy_score(y_test, preds), 4),
                        "Precision": round(
                            precision_score(
                                y_test, preds, zero_division=0
                            ),
                            4,
                        ),
                        "Recall": round(recall_score(y_test, preds), 4),
                        "F1-Score": round(
                            f1_score(y_test, preds, zero_division=0), 4
                        ),
                        "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
                        "CV F1 Mean": cv_score,
                    })
                    trained_models[name] = model
                    progress_bar.progress((idx + 1) / len(models))

                st.session_state.results = pd.DataFrame(results)
                st.session_state.trained_models = trained_models
                st.session_state.X_test = X_test_processed
                st.session_state.y_test = y_test

                st.success("⚡ Balanced models trained successfully!")
                st.dataframe(st.session_state.results, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: Performance Evaluation
# ---------------------------------------------------------
elif page == "5. Performance Evaluation":
    st.header("📈 5. Model Diagnostics & Performance Metrics")

    if "results" not in st.session_state:
        st.warning("⚠️ Please train models on Page 4 first.")
    else:
        selected_m = st.selectbox(
            "Select Model for Evaluation", st.session_state.results["Model"]
        )
        model = st.session_state.trained_models[selected_m]

        preds = model.predict(st.session_state.X_test)
        probs = model.predict_proba(st.session_state.X_test)[:, 1]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(st.session_state.y_test, preds)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=["#fef08a", "#c8a2c8", "#1e3a8a"],
                labels=dict(x="Predicted Label", y="True Label"),
            )
            fig_cm.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(st.session_state.y_test, probs)
            auc_val = roc_auc_score(st.session_state.y_test, probs)

            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"AUC = {auc_val:.3f}",
                    line=dict(color="#2563eb", width=3),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="#facc15"),
                )
            )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ---------------------------------------------------------
# PAGE 6: AI Retention Recommendation Engine
# ---------------------------------------------------------
elif page == "6. AI Retention Engine":
    st.header("💡 6. Retention Strategy Engine")

    if "trained_models" not in st.session_state:
        st.warning("⚠️ Please train models on Page 4 first.")
    else:
        col_m, col_i = st.columns([1, 1])

        with col_m:
            best_model_name = st.selectbox(
                "Select Active Model",
                list(st.session_state.trained_models.keys()),
            )
            best_model = st.session_state.trained_models[best_model_name]

        with col_i:
            max_idx = len(st.session_state.X_test) - 1
            sample_idx = st.number_input(
                f"Select Customer Index (0 to {max_idx})",
                min_value=0,
                max_value=max_idx,
                value=0,
            )

        sample_input = st.session_state.X_test[sample_idx].reshape(1, -1)
        prob = float(best_model.predict_proba(sample_input)[0][1])

        st.markdown("<br>", unsafe_allow_html=True)

        rc1, rc2 = st.columns([1, 2])

        with rc1:
            risk_color = (
                "#ef4444"
                if prob > 0.7
                else ("#eab308" if prob > 0.4 else "#16a34a")
            )
            risk_label = (
                "HIGH RISK"
                if prob > 0.7
                else ("MEDIUM RISK" if prob > 0.4 else "LOW RISK")
            )

            st.markdown(
                f"""
            <div class="metric-card" style="border-color: {risk_color};">
                <div class="metric-title">Predicted Risk Status</div>
                <div class="metric-value" style="color: {risk_color};">{risk_label}</div>
                <h2 style="color: #1e3a8a; margin-top: 10px;">{prob * 100:.1f}%</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with rc2:
            st.subheader("Recommended Action Plan")
            if prob > 0.7:
                st.error("🚨 **High Attrition Risk**")
                st.markdown("""
                * **Immediate Incentive:** Offer 20% discount on 12-month contract renewal.
                * **Support Elevation:** Priority routing to senior account specialists.
                * **Outreach:** Assign a dedicated success manager.
                """)
            elif prob > 0.4:
                st.warning("⚠️ **Moderate Attrition Risk**")
                st.markdown("""
                * **Engagement:** Trigger an automated satisfaction survey.
                * **Promotional Credit:** Offer 10% promotional credit toward service upgrades.
                """)
            else:
                st.success("✅ **Low Attrition Risk**")
                st.markdown("""
                * **Standard Engagement:** Keep customer on regular product updates list.
                * **Up-Sell Opportunity:** Recommend relevant service expansion options during billing cycles.
                """)