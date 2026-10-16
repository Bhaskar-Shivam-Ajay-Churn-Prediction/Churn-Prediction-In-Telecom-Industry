"""
app.py — Streamlit UI for the Telecom Customer Churn Prediction project.

Run with:
    streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    TARGET,
    df_to_excel_bytes,
    get_categorical_cols,
    get_numeric_cols,
    load_model,
    load_training_data,
    run_prediction,
    MODEL_PATH,
    ROOT,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e0ff !important; }

    /* Main background */
    .main { background-color: #0d0d1a; color: #e0e0ff; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d2b6b 100%);
        border: 1px solid #4a4a8a;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(80, 70, 200, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(100, 80, 255, 0.35);
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #9090cc;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #c4b5fd;
        margin-bottom: 0.5rem;
        padding-bottom: 8px;
        border-bottom: 2px solid #4a4a8a;
    }

    /* Prediction table highlight */
    .churn-yes { color: #f87171; font-weight: 600; }
    .churn-no  { color: #34d399; font-weight: 600; }

    /* Streamlit element overrides */
    div[data-testid="stMetricValue"] { color: #a78bfa !important; }
    .stSelectbox label, .stMultiSelect label { color: #c4b5fd !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #9090cc !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #a78bfa !important;
        border-bottom-color: #a78bfa !important;
    }

    /* Upload area */
    [data-testid="stFileUploadDropzone"] {
        background: #1a1a3e !important;
        border: 2px dashed #4a4a8a !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center; color:#a78bfa; font-size:1.5rem;'>📡 ChurnSense</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#9090cc; font-size:0.8rem;'>Telecom Churn Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠  Home", "📊  EDA Dashboard", "🤖  Predict"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<p style='color:#606090; font-size:0.75rem; text-align:center;'>Powered by scikit-learn & Streamlit</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Helper: plotly dark layout defaults
# ---------------------------------------------------------------------------
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,26,62,0.5)",
    font=dict(family="Inter", color="#c0c0e8"),
    xaxis=dict(gridcolor="#2a2a5a", linecolor="#2a2a5a"),
    yaxis=dict(gridcolor="#2a2a5a", linecolor="#2a2a5a"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c0c0e8")),
)

COLOR_SEQ = px.colors.sequential.Purpor

# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "🏠  Home":
    st.markdown("<h1 style='color:#a78bfa;'>📡 Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9090cc; font-size:1rem;'>An end-to-end ML pipeline built on the <strong style='color:#c4b5fd'>Databel</strong> dataset "
        "— explore patterns, train a model, and predict which customers are at risk of leaving.</p>",
        unsafe_allow_html=True,
    )

    df = load_training_data()
    total = len(df)
    churned = df[TARGET].astype(str).str.strip().str.lower().eq("yes").sum()
    churn_rate = churned / total * 100

    pipeline, feature_columns = load_model()
    model_status = "✅ Loaded" if pipeline is not None else "⚠️ Not trained"

    # --- KPI cards ---
    st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        (col1, f"{total:,}",         "Total Customers"),
        (col2, f"{churned:,}",       "Churned Customers"),
        (col3, f"{churn_rate:.1f}%", "Churn Rate"),
        (col4, model_status,         "Model Status"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-value'>{val}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Two charts side-by-side ---
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='section-header'>Churn Distribution</div>", unsafe_allow_html=True)
        counts = df[TARGET].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.6,
            marker=dict(colors=["#f87171", "#34d399"], line=dict(color="#0d0d1a", width=3)),
            textinfo="label+percent",
            textfont=dict(color="#e0e0ff", size=13),
        ))
        fig_donut.update_layout(
            **DARK_LAYOUT,
            showlegend=True,
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(text=f"<b>{churn_rate:.1f}%</b><br>Churn", x=0.5, y=0.5,
                               font=dict(size=16, color="#e0e0ff"), showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with right:
        st.markdown("<div class='section-header'>Churn by Contract Type</div>", unsafe_allow_html=True)
        if "Contract Type" in df.columns:
            ct = df.groupby(["Contract Type", TARGET]).size().reset_index(name="Count")
            fig_bar = px.bar(
                ct, x="Contract Type", y="Count", color=TARGET,
                color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
                barmode="group",
            )
            fig_bar.update_layout(**DARK_LAYOUT, height=320, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- Dataset preview ---
    st.markdown("<div class='section-header'>Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, height=280)

    # --- Model training section ---
    st.markdown("<div class='section-header'>Model Training</div>", unsafe_allow_html=True)
    if pipeline is None:
        st.warning("No trained model found. Click the button below to train the Logistic Regression pipeline.")
    else:
        st.success("A trained model is available. You can re-train to refresh it.")

    if st.button("🚀  Train / Re-train Model", type="primary"):
        with st.spinner("Training the model — this may take a minute…"):
            script = ROOT / "train_pipeline.py"
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Model trained successfully! Navigate to the **Predict** page to use it.")
            st.code(result.stdout, language="text")
            # Clear cached model so next load picks up new file
            load_model.clear()
        else:
            st.error("Training failed.")
            st.code(result.stderr, language="text")

# ============================================================
# PAGE 2 — EDA DASHBOARD
# ============================================================
elif page == "📊  EDA Dashboard":
    st.markdown("<h1 style='color:#a78bfa;'>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9090cc;'>Interactively explore the training dataset to understand churn patterns.</p>",
        unsafe_allow_html=True,
    )

    df = load_training_data()
    numeric_cols  = get_numeric_cols(df, exclude=[TARGET])
    cat_cols      = get_categorical_cols(df, exclude=[TARGET, "Customer ID", "Phone Number", "Churn Category", "Churn Reason"])

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Categorical Analysis", "📦 Distribution / Box", "🔥 Correlation Heatmap", "🗂 Raw Data"])

    # ---- Tab 1: Categorical bar chart ----
    with tab1:
        st.markdown("<div class='section-header'>Churn by Category</div>", unsafe_allow_html=True)
        col_sel, agg_sel = st.columns([3, 1])
        with col_sel:
            cat_feature = st.selectbox("Select a categorical feature", cat_cols, key="cat_feat")
        with agg_sel:
            normalize = st.checkbox("Show as %", value=False)

        if cat_feature:
            grouped = df.groupby([cat_feature, TARGET]).size().reset_index(name="Count")
            if normalize:
                totals = grouped.groupby(cat_feature)["Count"].transform("sum")
                grouped["Count"] = (grouped["Count"] / totals * 100).round(2)
                y_label = "Percentage (%)"
            else:
                y_label = "Count"

            fig = px.bar(
                grouped, x=cat_feature, y="Count", color=TARGET,
                color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
                barmode="group",
                labels={"Count": y_label},
                title=f"Churn by {cat_feature}",
            )
            fig.update_layout(**DARK_LAYOUT, height=420, title_font=dict(color="#c4b5fd", size=15))
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Extra: Payment Method pie
        if "Payment Method" in df.columns:
            st.markdown("<div class='section-header'>Payment Method Split</div>", unsafe_allow_html=True)
            pm = df["Payment Method"].value_counts().reset_index()
            pm.columns = ["Payment Method", "Count"]
            fig_pm = px.pie(
                pm, names="Payment Method", values="Count",
                color="Payment Method",
                color_discrete_map={
                    "Credit Card": "#38bdf8",
                    "Direct Debit": "#facc15",
                    "Paper Check": "#ef4444",
                },
                hole=0.45,
            )
            fig_pm.update_layout(**DARK_LAYOUT, height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_pm, use_container_width=True)

    # ---- Tab 2: Box / histogram ----
    with tab2:
        st.markdown("<div class='section-header'>Numeric Feature Distribution by Churn</div>", unsafe_allow_html=True)
        chart_type = st.radio("Chart type", ["Box Plot", "Histogram"], horizontal=True)
        num_feature = st.selectbox("Select a numeric feature", numeric_cols, key="num_feat")

        if num_feature:
            if chart_type == "Box Plot":
                fig2 = px.box(
                    df, x=TARGET, y=num_feature, color=TARGET,
                    color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
                    notched=True,
                    title=f"{num_feature} by Churn Status",
                )
            else:
                fig2 = px.histogram(
                    df, x=num_feature, color=TARGET, nbins=40,
                    color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
                    barmode="overlay", opacity=0.75,
                    title=f"Distribution of {num_feature}",
                )
            fig2.update_layout(**DARK_LAYOUT, height=420, title_font=dict(color="#c4b5fd", size=15))
            st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.markdown("<div class='section-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df[numeric_cols].describe().T.style.background_gradient(cmap="Purples"), use_container_width=True)

    # ---- Tab 3: Correlation heatmap ----
    with tab3:
        st.markdown("<div class='section-header'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
        corr_cols = st.multiselect(
            "Select numeric columns (max 15)",
            numeric_cols,
            default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols,
        )
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr().round(2)
            fig3 = px.imshow(
                corr,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                text_auto=True,
                title="Pearson Correlation Matrix",
            )
            fig3.update_layout(**DARK_LAYOUT, height=520, title_font=dict(color="#c4b5fd", size=15))
            fig3.update_traces(textfont=dict(size=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Select at least 2 numeric columns to display the heatmap.")

    # ---- Tab 4: Raw data ----
    with tab4:
        st.markdown("<div class='section-header'>Raw Training Data</div>", unsafe_allow_html=True)
        search = st.text_input("🔍 Filter by Customer ID or any text", "")
        display_df = df.copy()
        if search.strip():
            mask = display_df.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False)).any(axis=1)
            display_df = display_df[mask]

        st.caption(f"Showing {len(display_df):,} rows")
        st.dataframe(display_df, use_container_width=True, height=480)

        st.download_button(
            label="⬇️  Download Filtered Data as Excel",
            data=df_to_excel_bytes(display_df),
            file_name="filtered_training_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ============================================================
# PAGE 3 — PREDICT
# ============================================================
elif page == "🤖  Predict":
    st.markdown("<h1 style='color:#a78bfa;'>🤖 Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9090cc;'>Upload a new customer Excel file and get instant churn predictions.</p>",
        unsafe_allow_html=True,
    )

    pipeline, feature_columns = load_model()

    # ---- Model check ----
    if pipeline is None:
        st.error(
            "⚠️ No trained model found. Please go to the **🏠 Home** page and click **Train / Re-train Model** first."
        )
        st.stop()

    st.success(f"✅ Model loaded — {len(feature_columns)} features used during training.")

    # ---- File upload ----
    st.markdown("<div class='section-header'>Upload Customer Data</div>", unsafe_allow_html=True)
    upload_col, info_col = st.columns([2, 1])

    with upload_col:
        uploaded = st.file_uploader(
            "Drop an Excel file (.xlsx) of new customers",
            type=["xlsx"],
            help="The file should contain the same feature columns used during training.",
        )

    with info_col:
        with st.expander("ℹ️ What columns are needed?", expanded=False):
            st.markdown("The model expects the following **feature columns** (same as training):")
            st.code("\n".join(feature_columns), language="text")

    # ---- Use sample data if nothing uploaded ----
    sample_path = ROOT / "data" / "new_customers.xlsx"
    use_sample = False

    if uploaded is None:
        if sample_path.exists():
            st.info(
                "No file uploaded — using the bundled **data/new_customers.xlsx** as a demo. "
                "Upload your own file above to override."
            )
            use_sample = True
        else:
            st.warning("No file uploaded and no sample file found. Please upload an Excel file.")
            st.stop()

    # ---- Load data ----
    try:
        if use_sample:
            new_df = pd.read_excel(sample_path)
        else:
            new_df = pd.read_excel(uploaded)
        new_df.columns = new_df.columns.str.strip()
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        st.stop()

    if new_df.empty:
        st.warning("The uploaded file contains no rows.")
        st.stop()

    # ---- Run predictions ----
    with st.spinner("Running churn predictions…"):
        results = run_prediction(new_df, pipeline, feature_columns)

    total_customers  = len(results)
    churn_count      = (results["Predicted Churn"] == "Yes").sum()
    no_churn_count   = (results["Predicted Churn"] == "No").sum()
    avg_prob         = results["Churn Probability (%)"].mean()

    # ---- Summary KPIs ----
    st.markdown("<div class='section-header'>Prediction Summary</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    summary_kpis = [
        (c1, f"{total_customers:,}",    "Customers Processed"),
        (c2, f"{churn_count:,}",        "Predicted to Churn"),
        (c3, f"{no_churn_count:,}",     "Predicted to Stay"),
        (c4, f"{avg_prob:.1f}%",        "Avg Churn Probability"),
    ]
    for col, val, label in summary_kpis:
        with col:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-value'>{val}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Probability distribution chart ----
    chart_col, filter_col = st.columns([2, 1])
    with filter_col:
        filter_opt = st.radio("Filter results", ["All", "Churn (Yes)", "No Churn (No)"], index=0)
        threshold  = st.slider("Churn probability threshold (%)", 0, 100, 50, step=5)

    with chart_col:
        st.markdown("<div class='section-header'>Churn Probability Distribution</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            results, x="Churn Probability (%)", color="Predicted Churn",
            color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
            nbins=30, barmode="overlay", opacity=0.8,
            title="",
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#facc15",
                           annotation_text=f"Threshold: {threshold}%",
                           annotation_font_color="#facc15")
        fig_hist.update_layout(**DARK_LAYOUT, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---- Results table ----
    st.markdown("<div class='section-header'>Full Prediction Results</div>", unsafe_allow_html=True)

    display_results = results.copy()
    if filter_opt == "Churn (Yes)":
        display_results = display_results[display_results["Predicted Churn"] == "Yes"]
    elif filter_opt == "No Churn (No)":
        display_results = display_results[display_results["Predicted Churn"] == "No"]

    # Sort by probability descending so highest risk customers appear first
    display_results = display_results.sort_values("Churn Probability (%)", ascending=False).reset_index(drop=True)

    st.caption(f"Showing {len(display_results):,} of {total_customers:,} customers")
    st.dataframe(
        display_results.style.background_gradient(
            subset=["Churn Probability (%)"],
            cmap="RdYlGn_r",
            vmin=0, vmax=100,
        ),
        use_container_width=True,
        height=420,
    )

    # ---- Download ----
    st.download_button(
        label="⬇️  Download Predictions as Excel",
        data=df_to_excel_bytes(display_results),
        file_name="churn_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
