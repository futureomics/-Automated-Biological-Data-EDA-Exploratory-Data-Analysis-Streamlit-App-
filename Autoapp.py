# streamlit run app.py
# --- Requirements (install in your env) ---
# pip install streamlit pandas numpy plotly

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------
# App Config
# ----------------------
st.set_page_config(
    page_title="Automated Biological Data EDA",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Automated Biological Data EDA")
st.caption(
    "Upload a CSV to auto‑explore your dataset: histograms, scatter plots, box plots, correlations, and missingness overview."
)

# ----------------------
# Helpers
# ----------------------

def _safe_number_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _safe_cat_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, sep: str, encoding: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding, nrows=nrows)


def _maybe_log_transform(df: pd.DataFrame, cols: list[str], base: float = 10.0) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        mask_pos = df2[c] > 0
        with np.errstate(divide="ignore"):
            df2.loc[mask_pos, c] = np.log(df2.loc[mask_pos, c]) / np.log(base)
    return df2


# ----------------------
# Sidebar – Data input
# ----------------------
with st.sidebar:
    st.header("Data Input")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"]) 

    use_sample = st.checkbox("Use sample gene expression data", value=False)
    sep = st.selectbox("CSV delimiter", options=[",", ";", "\t", "|"], index=0)
    encoding = st.selectbox("Encoding", options=["utf-8", "latin-1", "utf-16"], index=0)
    nrows_opt = st.text_input("Read only first N rows (optional)")
    nrows = int(nrows_opt) if nrows_opt.strip().isdigit() else None

    st.divider()
    st.header("Global Options")
    sample_rows = st.slider("Sample rows for interactive plots (speeds up large data)", 200, 100_000, 10_000, step=200)
    enable_log = st.checkbox("Log10-transform numeric columns (useful for expression data)")

# ----------------------
# Load data
# ----------------------
if use_sample:
    rng = np.random.default_rng(42)
    genes = [f"Gene_{i:04d}" for i in range(1, 1001)]
    samples = [f"Sample_{j:02d}" for j in range(1, 13)]
    mat = rng.lognormal(mean=2.0, sigma=0.7, size=(len(genes), len(samples)))
    df_raw = pd.DataFrame(mat, columns=samples)
    df_raw.insert(0, "Gene", genes)
else:
    df_raw = None
    if uploaded is not None:
        try:
            df_raw = load_csv(uploaded.getvalue(), sep=sep, encoding=encoding, nrows=nrows)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

if df_raw is None:
    st.info("👆 Upload a CSV or toggle the sample data in the sidebar to begin.")
    st.stop()

orig_df = df_raw.copy()

empty_cols = [c for c in df_raw.columns if df_raw[c].isna().all()]
if empty_cols:
    df_raw = df_raw.drop(columns=empty_cols)

df_raw = df_raw.replace({"NA": np.nan, "NaN": np.nan, "None": np.nan})

if enable_log:
    num_cols = _safe_number_cols(df_raw)
    df_log = _maybe_log_transform(df_raw, num_cols, base=10.0)
    df = df_log
else:
    df = df_raw

with st.expander("🔎 Data Preview", expanded=True):
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df.head(50), use_container_width=True)
    with right:
        st.markdown("**Shape**")
        st.write(df.shape)
        st.markdown("**Columns**")
        st.write(list(df.columns))
        st.markdown("**Missing values (per column)**")
        st.write(df.isna().sum().sort_values(ascending=False))

plot_df = df.sample(min(sample_rows, len(df)), random_state=42) if len(df) > sample_rows else df

num_cols = _safe_number_cols(plot_df)
cat_cols = _safe_cat_cols(plot_df)

# ----------------------
# Univariate – Histograms & Box
# ----------------------
st.markdown("### Univariate Distributions")
col1, col2 = st.columns(2)
with col1:
    if num_cols:
        col_sel = st.selectbox("Histogram – numeric column", options=num_cols)
        bins = st.slider("Bins", 10, 200, 50)
        fig = px.histogram(plot_df, x=col_sel, nbins=bins, marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns detected for histograms.")

with col2:
    if cat_cols:
        cat_sel = st.selectbox("Bar chart – categorical column", options=cat_cols)
        counts = plot_df[cat_sel].astype("category").value_counts().reset_index()
        counts.columns = [cat_sel, "count"]
        fig = px.bar(counts, x=cat_sel, y="count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns detected for bar charts.")

# ----------------------
# Bivariate – Scatter & Box by Group
# ----------------------
st.markdown("### Bivariate Relationships")
col3, col4 = st.columns(2)

with col3:
    if len(num_cols) >= 2:
        x_sel = st.selectbox("Scatter: X", options=num_cols, key="scatter_x")
        y_sel = st.selectbox("Scatter: Y", options=[c for c in num_cols if c != x_sel], key="scatter_y")
        color_sel = st.selectbox("Color (optional)", options=[None] + cat_cols, index=0)
        fig = px.scatter(plot_df, x=x_sel, y=y_sel, color=color_sel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for scatter plots.")

with col4:
    if num_cols and cat_cols:
        y_num = st.selectbox("Box plot: numeric", options=num_cols)
        x_cat = st.selectbox("Box plot: group by", options=cat_cols)
        fig = px.box(plot_df, x=x_cat, y=y_num, points="outliers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need one numeric and one categorical column for grouped box plots.")

# ----------------------
# Correlation Matrix
# ----------------------
st.markdown("### Correlation Matrix")
if len(num_cols) >= 2:
    method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)
    corr = plot_df[num_cols].corr(method=method)
    fig = px.imshow(corr, text_auto=False, aspect="auto", origin="lower", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need at least two numeric columns to compute correlations.")

# ----------------------
# Missingness Overview
# ----------------------
st.markdown("### Missing Values")
miss = df.isna().mean().sort_values(ascending=False)
miss_df = miss.reset_index()
miss_df.columns = ["column", "missing_fraction"]
fig = px.bar(miss_df, x="column", y="missing_fraction")
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Footer
# ----------------------
st.divider()
st.markdown()
st.caption("Made with ❤️ using Streamlit and Plotly · By Future Omics 🤖Bioinformatics made easy ❤️")