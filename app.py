# app.py (modified: inject feature-engineering + clustering but keep design unchanged)
# Fixed: robust single-employee clustering fallback + ensure cluster_map keys are ints

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import joblib
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# === DEFINE CSS FIRST (WAJIB) ===
table_css = """
<style>

/* Background keseluruhan tabel */
[data-testid="stDataFrame"] .st-ag-theme-streamlit-light {
    background-color: #2e307d !important;
}

/* Cell background */
[data-testid="stDataFrame"] .ag-root-wrapper,
[data-testid="stDataFrame"] .ag-center-cols-container,
[data-testid="stDataFrame"] .ag-cell {
    background-color: #2e307d !important;
    color: white !important;
}

/* Header background */
[data-testid="stDataFrame"] .ag-header,
[data-testid="stDataFrame"] .ag-header-cell-label {
    background-color: #1f225a !important;
    color: white !important;
    font-weight: 600 !important;
}

/* Border clean look */
[data-testid="stDataFrame"] .ag-row,
[data-testid="stDataFrame"] .ag-cell {
    border: none !important;
}

</style>
"""

st.markdown(table_css, unsafe_allow_html=True)

# === PAGE BACKGROUND ===
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #449fe3;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# Helper: feature engineering + clustering
# ---------------------------
def apply_feature_engineering_and_clustering(df_in,
                                             cluster_model_path="cluster_model.pkl",
                                             cluster_scaler_path="cluster_scaler.pkl",
                                             cluster_meta_path="cluster_metadata.json",
                                             force_recompute=False):
    """
    Input: dataframe with columns:
        Employee_ID, Age, Performance_Score, Leadership_Score, Training_Hours,
        Projects_Handled, Peer_Review_Score, Current_Position_Level, Salary (opt)
    Output: same dataframe with added calculated columns:
        Leadership_Index, Performance_Index, Potential_Index,
        Projects_Handled_scaled, Training_Hours_scaled,
        Performance_Consistency, Growth_Momentum, Cluster (+ metadata)
    Behavior:
        - If pre-trained scaler + kmeans exist and force_recompute==False -> load and use them
        - Else fit scaler + kmeans on provided df and save them
    """
    df = df_in.copy()

    # ensure required raw columns exist
    required_raw = {
        "Performance_Score", "Leadership_Score", "Training_Hours",
        "Projects_Handled", "Peer_Review_Score"
    }
    if not required_raw.issubset(set(df.columns)):
        # we cannot proceed if core inputs missing
        return df

    # safe numeric conversion
    for c in ["Performance_Score","Leadership_Score","Training_Hours","Projects_Handled","Peer_Review_Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # INDEX calculations
    df["Leadership_Index"] = 0.4 * df["Leadership_Score"] + 0.6 * df["Peer_Review_Score"]
    df["Performance_Index"] = (
        0.5 * df["Performance_Score"] + 0.2 * df["Projects_Handled"] + 0.3 * df["Peer_Review_Score"]
    )
    df["Potential_Index"] = (
        0.4 * df["Training_Hours"] + 0.4 * df["Peer_Review_Score"] + 0.2 * df["Leadership_Score"]
    )

    # Scale Projects_Handled and Training_Hours for the two derived metrics
    scaler_cols = ["Projects_Handled", "Training_Hours"]
    scaler = None

    if (os.path.exists(cluster_scaler_path) and not force_recompute):
        try:
            scaler = joblib.load(cluster_scaler_path)
        except Exception:
            scaler = None

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[scaler_cols].values)
        try:
            joblib.dump(scaler, cluster_scaler_path)
        except Exception:
            pass

    scaled_vals = scaler.transform(df[scaler_cols].values)
    df["Projects_Handled_scaled"] = scaled_vals[:, 0]
    df["Training_Hours_scaled"] = scaled_vals[:, 1]

    # Derived metrics per your formulas
    df["Performance_Consistency"] = df["Performance_Score"] * df["Projects_Handled_scaled"]
    df["Growth_Momentum"] = df["Projects_Handled_scaled"] / (df["Training_Hours_scaled"] + 1.0)

    # CLUSTERING: use Performance_Index, Leadership_Index, Potential_Index
    cluster_features = ["Performance_Index","Leadership_Index","Potential_Index"]
    X_cluster = df[cluster_features].values

    # scaler for clustering (standardize the 3 indices)
    cluster_scaler_for_3 = None
    cluster_scaler_3_path = cluster_scaler_path.replace(".pkl","_3.pkl")
    if (os.path.exists(cluster_scaler_3_path) and not force_recompute):
        try:
            cluster_scaler_for_3 = joblib.load(cluster_scaler_3_path)
        except Exception:
            cluster_scaler_for_3 = None

    if cluster_scaler_for_3 is None:
        cluster_scaler_for_3 = StandardScaler()
        cluster_scaler_for_3.fit(X_cluster)
        try:
            joblib.dump(cluster_scaler_for_3, cluster_scaler_3_path)
        except Exception:
            pass

    X_cluster_scaled = cluster_scaler_for_3.transform(X_cluster)

    # load or fit kmeans
    kmeans = None
    if (os.path.exists(cluster_model_path) and not force_recompute):
        try:
            kmeans = joblib.load(cluster_model_path)
        except Exception:
            kmeans = None

    if kmeans is None:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        try:
            joblib.dump(kmeans, cluster_model_path)
        except Exception:
            pass

    # assign cluster (1..4)
    labels = kmeans.predict(X_cluster_scaled)
    df["Cluster"] = labels + 1

    # try to load cluster metadata json, else create default mapping
    cluster_meta = None
    if os.path.exists(cluster_meta_path):
        try:
            with open(cluster_meta_path, "r") as f:
                cluster_meta = json.load(f)
        except Exception:
            cluster_meta = None

    if cluster_meta is None:
        # default metadata (same semantics you provided earlier)
        cluster_meta = {
            "Characteristics": {
                "1": "Under Developed With Potential",
                "2": "At-Risk and Underpowered",
                "3": "All Around Top Performer",
                "4": "Consistent Performer or Leader"
            },
            "Description": {
                "1": "Talents in this cluster show strong potential...",
                "2": "This cluster has the lowest scores across all indices...",
                "3": "This group demonstrates the highest levels of performance...",
                "4": "Talents in this cluster excel in performance and execution..."
            },
            "HR_Recommendations": {
                "1": "Strengthen their acceleration potential...",
                "2": "Implement targeted capability recovery...",
                "3": "Fast-track their development...",
                "4": "Maximize their contribution through specialist paths..."
            },
            "HR_Programs": {
                "1": "Performance improvement training, coaching...",
                "2": "Core competency training, SOP refreshers...",
                "3": "Leadership bootcamps, strategic rotations...",
                "4": "Technical certifications, specialist pathways..."
            }
        }
        # attempt to persist so later runs use it
        try:
            with open(cluster_meta_path, "w") as f:
                json.dump(cluster_meta, f, indent=4)
        except Exception:
            pass

    # map metadata to df
    def map_meta(row, meta_dict, field):
        key = str(int(row["Cluster"]))
        return meta_dict.get(field, {}).get(key, "—")

    df["Characteristics"] = df.apply(lambda r: map_meta(r, cluster_meta, "Characteristics"), axis=1)
    df["Description"] = df.apply(lambda r: map_meta(r, cluster_meta, "Description"), axis=1)
    df["HR_Recommendations"] = df.apply(lambda r: map_meta(r, cluster_meta, "HR_Recommendations"), axis=1)
    df["HR_Programs"] = df.apply(lambda r: map_meta(r, cluster_meta, "HR_Programs"), axis=1)

    # done
    return df

# ---------------------------
# Load dataset (original path)
# ---------------------------
DATA_PATH = "data/Clean/dataset_clustered_dashboard.csv"
FALLBACK_RAW_PATH = "data/Clean/dataset_test_cluster.csv"

# prefer original clustered dataset if available, else load raw fallback and compute
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        # If it lacks any of the derived columns, try to enrich
        needed_cols = {"Performance_Index","Leadership_Index","Potential_Index","Cluster"}
        if not needed_cols.issubset(set(df.columns)):
            df = apply_feature_engineering_and_clustering(df)
    except Exception:
        # fallback to test dataset
        df = pd.read_csv(FALLBACK_RAW_PATH) if os.path.exists(FALLBACK_RAW_PATH) else pd.DataFrame()
        if not df.empty:
            df = apply_feature_engineering_and_clustering(df)
else:
    # try fallback raw
    if os.path.exists(FALLBACK_RAW_PATH):
        df = pd.read_csv(FALLBACK_RAW_PATH)
        df = apply_feature_engineering_and_clustering(df)
    else:
        df = pd.DataFrame()

# initialize session master_df if not present
if "master_df" not in st.session_state:
    st.session_state["master_df"] = df.copy()

# =========================
# HEADER (Tabs pindah ke bawah bagian ini)
# =========================
st.markdown('# **Rakamin HR Intelligence Hub**')
st.write('### *A centralized view of Rakamin workforce performance and potential.*')
st.write('*Created by Syntax Society*')    

# =========================
# CREATE TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Talent Overview", "Talent Performance", "Talent Predictor"])

# ======================================================================
# TAB 1 — TALENT OVERVIEW
# ======================================================================
with tab1:

    st.markdown("## 📸 Talent Overview")

    colA, colB = st.columns(2)

    def overview_card(title, value):
        return f"""
            <div style="
                border: 3px solid #2e307d;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 15px;
                background-color: #2e307d;
            ">
                <h3 style="margin:0;padding:0;color:white;font-size:22px;">{title}</h3>
                <div style="margin-top:10px;">
                    <span style="font-size:28px;color:white;">{value}</span>
                </div>
            </div>
            """

    # ---- COLUMN A ----
    with colA:
        total_talent = st.session_state["master_df"]["Employee_ID"].nunique() if not st.session_state["master_df"].empty else 0
        colA.markdown(overview_card("Total Talent", f"{total_talent}"), unsafe_allow_html=True)

    # ---- COLUMN B ----
    with colB:
        avg_age = st.session_state["master_df"]["Age"].mean() if not st.session_state["master_df"].empty else 0.0
        colB.markdown(overview_card("Average Talent Age", f"{avg_age:.1f} years"), unsafe_allow_html=True)

    # ============================
    # POSITION LEVEL CARD
    # ============================
    df_for_vis = st.session_state["master_df"].copy()

    level_order = ["Junior", "Mid", "Senior", "Lead"]

    count_by_level = (
        df_for_vis.groupby("Current_Position_Level")["Employee_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"Employee_ID": "Total_Talent"})
    )

    count_by_level = count_by_level[count_by_level["Current_Position_Level"].isin(level_order)]

    count_by_level["Current_Position_Level"] = pd.Categorical(
        count_by_level["Current_Position_Level"],
        categories=level_order,
        ordered=True
    )

    count_by_level = count_by_level.sort_values("Current_Position_Level")

    rows_html = ""
    for _, r in count_by_level.iterrows():
        rows_html += f"""
        <div style="display:flex;justify-content:space-between;padding:4px 0;
                    border-bottom:1px solid rgba(255,255,255,0.05);">
            <div style="font-size:22px;font-weight:600;color:white;">{r['Current_Position_Level']}</div>
            <div style="font-size:22px;color:#ddd;">{int(r['Total_Talent'])} talent</div>
        </div>
        """

    st.markdown(
        f"""
        <div style="width:100%;border:3px solid #2e307d;border-radius:12px;
                    padding:20px;margin-top:10px;background-color:#2e307d;">
            <div style="font-size:22px;font-weight:700;margin-bottom:12px;color:white;">
                Talent Count by Position Level
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================
    # Utility: fig → base64
    # ============================
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ============================
    # HISTOGRAM CARDS
    # ============================
    col1, col2 = st.columns(2)

    # --- AGE DISTRIBUTION ---
    with col1:
        fig, ax = plt.subplots(figsize=(5,3))
        if not df_for_vis.empty:
            sns.histplot(df_for_vis["Age"].dropna(), bins=12, color="#00bf63", ax=ax)
        ax.set_xlabel("Age", color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#2e307d")
        ax.set_facecolor("#2e307d")

        st.markdown(
            f"""
            <div style="border:3px solid #2e307d;border-radius:12px;
                        padding:20px;margin-top:20px;background-color:#2e307d;">
                <img src="data:image/png;base64,{fig_to_base64(fig)}" style="width:100%;border-radius:10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
        plt.close(fig)

    # --- POSITION LEVEL ---
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        if not df_for_vis.empty:
            sns.countplot(
                data=df_for_vis,
                x="Current_Position_Level",
                order=level_order,
                color="#00bf63",
                ax=ax2
            )
        ax2.set_xlabel("Position Level", color="white")
        ax2.set_ylabel("Count", color="white")
        ax2.tick_params(colors="white")
        fig2.patch.set_facecolor("#2e307d")
        ax2.set_facecolor("#2e307d")

        st.markdown(
            f"""
            <div style="border:3px solid #2e307d;border-radius:12px;
                        padding:20px;margin-top:20px;background-color:#2e307d;">
                <img src="data:image/png;base64,{fig_to_base64(fig2)}" style="width:100%;border-radius:10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
        plt.close(fig2)

    # ============================================
    # PERFORMANCE GROUP CARD (FIXED)
    # ============================================
    df_for_counts = df_for_vis if not df_for_vis.empty else pd.DataFrame(columns=["Cluster"])
    cluster_low = int(df_for_counts[df_for_counts["Cluster"] == 2].shape[0]) if "Cluster" in df_for_counts.columns else 0
    cluster_high = int(df_for_counts[df_for_counts["Cluster"] == 4].shape[0]) if "Cluster" in df_for_counts.columns else 0
    cluster_avg = int(df_for_counts[df_for_counts["Cluster"].isin([1,3])].shape[0]) if "Cluster" in df_for_counts.columns else 0

    cluster_rows = f"""
    <div style='display:flex; justify-content:space-between; padding:6px 0;
                border-bottom:1px solid rgba(255,255,255,0.08);'>
        <div style='font-size:22px; font-weight:600; color:white;'>Low Performing Talent</div>
        <div style='font-size:22px; font-weight:700; color:#ff5757;'>{cluster_low}</div>
    </div>

    <div style='display:flex; justify-content:space-between; padding:6px 0;
                border-bottom:1px solid rgba(255,255,255,0.08);'>
        <div style='font-size:22px; font-weight:600; color:white;'>High Performing Talent</div>
        <div style='font-size:22px; font-weight:700; color:#00bf63;'>{cluster_high}</div>
    </div>

    <div style='display:flex; justify-content:space-between; padding:6px 0;'>
        <div style='font-size:22px; font-weight:600; color:white;'>Average Talent</div>
        <div style='font-size:22px; color:#dddddd;'>{cluster_avg}</div>
    </div>
    """

    cluster_card = f"""
    <div style='width:100%; border:3px solid #2e307d; border-radius:12px;
                padding:20px; margin-top:15px; background-color:#2e307d;'>
        <div style='font-size:22px; font-weight:700; margin-bottom:12px; color:white;'>
            Talent Count by Performance Group
        </div>
        {cluster_rows}
    </div>
    """

    st.markdown(cluster_card, unsafe_allow_html=True)

    # ============================
    # POTENTIAL LOSS CARD
    # ============================
    salary_col = "Salary"
    if salary_col not in df_for_vis.columns:
        st.markdown(
            "<div style='width:100%; border:3px solid #2e307d; border-radius:12px;"
            " padding:20px; margin-top:15px; background-color:#2e307d;"
            " font-family:Inter, sans-serif;'>"
            "<div style='font-size:22px; font-weight:700; margin-bottom:12px; color:white;'>"
            "Potential Loss"
            "</div>"
            "<div style='font-size:16px; color:#dddddd;'>"
            "Column <strong>Salary</strong> was not found in the dataset."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        s = pd.to_numeric(df_for_vis.loc[df_for_vis["Cluster"] == 2, salary_col], errors="coerce")
        total_salary = s.sum(skipna=True)
        try:
            formatted_salary = "Rp " + "{:,.0f}".format(total_salary).replace(",", ".")
        except:
            formatted_salary = "Rp " + str(total_salary)

        html_card = (
            "<div style='width:100%; border:3px solid #2e307d; border-radius:12px;"
            " padding:20px; margin-top:15px; background-color:#2e307d;"
            " font-family:Inter, sans-serif;'>"

            "<div style='font-size:22px; font-weight:700; margin-bottom:12px; color:white;'>"
            "Potential Loss"
            "</div>"

            "<div style='display:flex; justify-content:space-between; align-items:center; padding:8px 0;'>"
            "<div style='font-size:16px; color:#dddddd;'>"
            "Total monthly salary paid to low-performing talent"
            "</div>"
            f"<div style='font-size:28px; font-weight:700; color:#ff5757;'>{formatted_salary}</div>"
            "</div>"

            "<div style='font-size:13px; margin-top:10px; color:#cfcfcf;'>"
            "This represents the estimated monthly salary load associated with low-performing talent — "
            "a direct indicator of potential productivity loss from a business perspective."
            "</div>"

            "</div>"
        )

        st.markdown(html_card, unsafe_allow_html=True)


# ======================================================================
# TAB 2 — TALENT PERFORMANCE (ASLI)
# ======================================================================
with tab2:

    # ⭐ Top Talent
    st.markdown("## ⭐ Top Talent")

    colA, colB = st.columns(2)

    with colA:
        category = st.selectbox(
            'Select ranking category:',
            ['Best Performing', 'Best Leadership', 'Best Potential'],
            key="ranking_category"
        )

    with colB:
        position_options = ['All Levels'] + sorted(st.session_state["master_df"]['Current_Position_Level'].dropna().unique().tolist()) if not st.session_state["master_df"].empty else ['All Levels']
        selected_level = st.selectbox(
            'Filter by Position Level:',
            position_options,
            key="position_filter"
        )

    df_filtered = st.session_state["master_df"] if selected_level == 'All Levels' else st.session_state["master_df"][st.session_state["master_df"]['Current_Position_Level'] == selected_level]

    if category == 'Best Performing':
        ranked = df_filtered.sort_values('Performance_Index', ascending=False).head(10) if 'Performance_Index' in df_filtered.columns else pd.DataFrame()
        st.dataframe(
            ranked[
                ['Employee_ID', 'Current_Position_Level', 'Performance_Index',
                 'Performance_Consistency', 'Cluster']
            ] if not ranked.empty else ranked,
            hide_index=True,
            use_container_width=True
        )

    elif category == 'Best Leadership':
        ranked = df_filtered.sort_values('Leadership_Index', ascending=False).head(10) if 'Leadership_Index' in df_filtered.columns else pd.DataFrame()
        st.dataframe(
            ranked[
                ['Employee_ID', 'Current_Position_Level', 'Leadership_Index',
                 'Leadership_Influence', 'Peer_Review_Score']
            ] if not ranked.empty else ranked,
            hide_index=True,
            use_container_width=True
        )

    elif category == 'Best Potential':
        ranked = df_filtered.sort_values('Potential_Index', ascending=False).head(10) if 'Potential_Index' in df_filtered.columns else pd.DataFrame()
        st.dataframe(
            ranked[
                ['Employee_ID', 'Current_Position_Level', 'Potential_Index',
                 'Growth_Momentum', 'Training_Hours']
            ] if not ranked.empty else ranked,
            hide_index=True,
            use_container_width=True
        )

    # ⚖️ Average Indexes
    st.markdown("## ⚖️ Average Indexes")

    col1, col2, col3 = st.columns(3)

    def metric_card(title, value):
        return f"""
        <div style="
            border: 3px solid #2e307d;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #2e307d;
        ">
            <h3 style="margin:0; padding:0; color:white; font-size:22px;">
                {title}
            </h3>
            <div style="margin-top:10px;">
                <span style="font-size:26px; font-weight:300; color:white;">
                    {value}
                </span>
            </div>
        </div>
        """

    perf_mean = st.session_state["master_df"]["Performance_Index"].mean() if 'Performance_Index' in st.session_state["master_df"].columns else 0.0
    lead_mean = st.session_state["master_df"]["Leadership_Index"].mean() if 'Leadership_Index' in st.session_state["master_df"].columns else 0.0
    pot_mean  = st.session_state["master_df"]["Potential_Index"].mean() if 'Potential_Index' in st.session_state["master_df"].columns else 0.0

    col1.markdown(metric_card('Performance Idx', f'{perf_mean:.2f}'), unsafe_allow_html=True)
    col2.markdown(metric_card('Leadership Idx', f'{lead_mean:.2f}'), unsafe_allow_html=True)
    col3.markdown(metric_card('Potential Idx', f'{pot_mean:.2f}'), unsafe_allow_html=True)

    # ⚠️ High Risk Talent
    st.markdown("## ⚠️ High Risk Talent")

    colR1, colR2 = st.columns(2)

    with colR1:
        risk_category = st.selectbox(
            'Select ranking category:',
            ['Low Performing', 'Low Leadership', 'Low Potential'],
            key="risk_category"
        )

    with colR2:
        position_options_risk = ['All Levels'] + sorted(st.session_state["master_df"]['Current_Position_Level'].dropna().unique().tolist()) if not st.session_state["master_df"].empty else ['All Levels']
        selected_level_risk = st.selectbox(
            'Filter by Position Level:',
            position_options_risk,
            key="risk_position_filter"
        )

    df_risk = st.session_state["master_df"] if selected_level_risk == 'All Levels' else st.session_state["master_df"][st.session_state["master_df"]['Current_Position_Level'] == selected_level_risk]

    if risk_category == 'Low Performing':
        ranked = df_risk.sort_values('Performance_Index', ascending=True).head(10) if 'Performance_Index' in df_risk.columns else pd.DataFrame()
        st.dataframe(
            ranked[
                ['Employee_ID','Current_Position_Level','Performance_Index',
                 'Performance_Consistency','Cluster']
            ] if not ranked.empty else ranked,
            hide_index=True, use_container_width=True
        )

    elif risk_category == 'Low Leadership':
        ranked = df_risk.sort_values('Leadership_Index', ascending=True).head(10) if 'Leadership_Index' in df_risk.columns else pd.DataFrame()
        st.dataframe(
            ranked[
                ['Employee_ID','Current_Position_Level','Leadership_Index',
                 'Leadership_Influence','Peer_Review_Score']
            ] if not ranked.empty else ranked,
            hide_index=True, use_container_width=True
        )

    else:
        ranked = df_risk.sort_values('Potential_Index', ascending=True).head(10) if 'Potential_Index' in df_risk.columns else pd.DataFrame()
        st.dataframe(
            ranked[
                ['Employee_ID','Current_Position_Level','Potential_Index',
                 'Growth_Momentum','Training_Hours']
            ] if not ranked.empty else ranked,
            hide_index=True, use_container_width=True
        )


# ======================================================================
# TAB 3 — TALENT PREDICTOR (ASLI)
# ======================================================================
with tab3:

    df_ref = st.session_state["master_df"]

    def small_card(title, value, color="white", bg="#2e307d"):
        return f"""
        <div style='border:3px solid #2e307d;border-radius:12px;
                    padding:16px;margin-bottom:15px;background-color:{bg};'>
            <h3 style='color:white;font-size:20px;margin:0'>{title}</h3>
            <div style='margin-top:12px;font-size:22px;color:{color}'>{value}</div>
        </div>
        """

    def build_description_card(text):
        return f"""
        <div style='border:3px solid #2e307d;border-radius:12px;
                    padding:20px;margin-top:20px;background-color:#2e307d;'>
            <div style='font-size:22px;font-weight:700;margin-bottom:12px;color:white'>Description</div>
            <div style='font-size:18px;color:#ddd'>{text}</div>
        </div>
        """

    def long_card(title, content):
        return f"""
        <div style='border:3px solid #2e307d;border-radius:12px;
                    padding:20px;margin-top:20px;background-color:#00bf63;'>
            <div style='font-size:22px;font-weight:700;color:white;margin-bottom:12px'>{title}</div>
            <div style='font-size:18px;color:white'>{content}</div>
        </div>
        """

    st.markdown("## 🔎 Talent Predictor")
    st.markdown("### Select Talent Input Method")

    # --- build cluster_map with int keys (robust)
    if {"Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"}.issubset(df_ref.columns):
        raw_map = (
            df_ref[["Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"]]
            .drop_duplicates("Cluster")
            .set_index("Cluster")
            .to_dict("index")
        )
        # ensure keys are ints
        cluster_map = {}
        for k, v in raw_map.items():
            try:
                ik = int(k)
            except Exception:
                try:
                    ik = int(float(k))
                except Exception:
                    continue
            cluster_map[ik] = v
    else:
        cluster_map = {}

    mode = st.radio(
        "",
        [
            "Select employee ID",
            "Predict employee cluster and characteristics",
            "Upload employee data in bulk using CSV"
        ],
        horizontal=True
    )

    emp = None
    is_empty = False

    # -----------------------------
    # 1) SELECT EMPLOYEE ID
    # -----------------------------
    if mode == "Select employee ID":
        st.markdown("#### Select Current Employee")

        colA, colB = st.columns(2)
        with colA:
            ids = list(df_ref["Employee_ID"].dropna().astype(str).unique()) if not df_ref.empty else []
            dropdown_vals = ["None"] + ids
            emp_dropdown = st.selectbox("Select Employee ID:", dropdown_vals, key="tp_dropdown")

        with colB:
            typed_id = st.text_input("Or type Employee ID:", placeholder="e.g. EMP0057", key="tp_typed_entry")

        if typed_id.strip() and typed_id.strip() in df_ref.get("Employee_ID", pd.Series(dtype=str)).astype(str).values:
            emp_id = typed_id.strip()
            try:
                st.session_state["tp_dropdown"] = "None"
            except Exception:
                pass
        else:
            emp_id = emp_dropdown

        if emp_id == "None" or emp_id is None:
            is_empty = True
            emp = {}
        else:
            row = df_ref[df_ref["Employee_ID"].astype(str) == str(emp_id)]
            if len(row) == 0:
                is_empty = True
                emp = {}
            else:
                emp = row.iloc[0].to_dict()

    # -----------------------------
    # 2) PREDICT EMPLOYEE (MANUAL FORM)
    # -----------------------------
    elif mode == "Predict employee cluster and characteristics":
        st.markdown("### Predict Employee Cluster and Characteristics")

        emp_id_input = st.text_input("Employee ID (optional):", placeholder="e.g. NEW001", key="m_empid")

        age_raw = st.text_input("Age", placeholder="Enter age (18–70)", key="m_age")
        try:
            age = int(age_raw) if age_raw.strip() else None
            if age is not None and not (18 <= age <= 70):
                age = None
        except:
            age = None

        perf = st.selectbox(
            "Performance Score (1–5)",
            ["Choose an option"] + list(range(1,6)),
            index=0,
            key="m_perf"
        )

        lead_options = ["Choose an option"] + list(range(0, 101, 10))
        lead = st.selectbox(
            "Leadership Score (0–100)",
            lead_options,
            index=0,
            key="m_lead"
        )

        train_options = ["Choose an option"] + list(range(0, 201, 20))
        train = st.selectbox(
            "Training Hours (0–200)",
            train_options,
            index=0,
            key="m_train"
        )

        proj_options = ["Choose an option"] + list(range(0, 21))
        proj = st.selectbox(
            "Projects Handled (0–20)",
            proj_options,
            index=0,
            key="m_proj"
        )

        peer_options = ["Choose an option"] + list(range(0, 101, 10))
        peer = st.selectbox(
            "Peer Review Score (0–100)",
            peer_options,
            index=0,
            key="m_peer"
        )

        level = st.selectbox(
            "Current Position Level",
            ["Choose an option","Junior","Mid","Senior","Lead"],
            index=0,
            key="m_level"
        )

        required_filled = (
            (age is not None) and
            (perf != "Choose an option") and
            (lead != "Choose an option") and
            (train != "Choose an option") and
            (proj != "Choose an option") and
            (peer != "Choose an option") and
            (level != "Choose an option")
        )

        if not required_filled:
            is_empty = True
            emp = {}
        else:
            is_empty = False
            emp = {
                "Employee_ID": emp_id_input.strip() if emp_id_input.strip() else "—",
                "Age": age,
                "Performance_Score": int(perf),
                "Leadership_Score": float(lead),
                "Training_Hours": float(train),
                "Projects_Handled": float(proj),
                "Peer_Review_Score": float(peer),
                "Current_Position_Level": level,
            }

        # -----------------------------
    # 3) UPLOAD CSV
    # -----------------------------
    elif mode == "Upload employee data in bulk using CSV":
        st.markdown("*Upload a CSV following the required format.*")
        st.markdown('<a id="select-employee-id"></a>', unsafe_allow_html=True)

        template_df = pd.DataFrame({
            "Employee_ID":["EMP0001"],
            "Age":[30],
            "Performance_Score":[5],
            "Leadership_Score":[60],
            "Training_Hours":[40],
            "Projects_Handled":[5],
            "Peer_Review_Score":[75],
            "Current_Position_Level":["Senior"]
        })

        st.download_button("Download template CSV", template_df.to_csv(index=False), "employee_template.csv")
        uploaded = st.file_uploader("Upload CSV:", type=["csv"])

        if uploaded:
            try:
                new = pd.read_csv(uploaded)

                required_cols = {
                    "Employee_ID","Age","Performance_Score","Leadership_Score",
                    "Training_Hours","Projects_Handled","Peer_Review_Score","Current_Position_Level"
                }

                if not required_cols.issubset(set(new.columns)):
                    # Big red card (consistent style)
                    st.markdown(
                        """
                        <div style="
                            margin-top:18px;
                            padding:22px;
                            border-radius:14px;
                            background-color:#ff4d4f;
                            color:white;
                            font-size:19px;
                            font-weight:700;
                            line-height:1.5;
                        ">
                            ✗ Upload failed.<br>
                            Required columns are missing. Please use the template.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # --- SAFE: enrich using combined dataframe so clustering/training won't run on tiny 'new' only ---
                    existing = st.session_state.get("master_df", pd.DataFrame())
                    # If master_df is empty, just concat so apply_* sees more samples if available in existing;
                    # otherwise apply to combined to let function use existing centroids/models.
                    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new.copy()

                    # Do NOT force recompute (we prefer using saved models if present).
                    combined_enriched = apply_feature_engineering_and_clustering(combined, force_recompute=False)

                    # Extract only the newly uploaded rows (they are at the tail).
                    new_enriched = combined_enriched.iloc[len(combined_enriched) - len(new):].reset_index(drop=True)

                    # Append enriched new rows to master_df
                    if existing.empty:
                        st.session_state["master_df"] = new_enriched.copy()
                    else:
                        st.session_state["master_df"] = pd.concat([existing, new_enriched], ignore_index=True)

                    # Refresh local ref
                    df_ref = st.session_state["master_df"]

                    # === BIG SUCCESS GREEN CARD (stand-out, clickable link) ===
                    st.markdown(
                        f"""
                        <div style="
                            margin-top:18px;
                            padding:22px;
                            border-radius:14px;
                            background: linear-gradient(180deg,#0fb35f,#00bf63);
                            color:white;
                            font-size:20px;
                            font-weight:700;
                            line-height:1.4;
                            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
                        ">
                            <div style="display:flex;align-items:center;gap:18px;">
                                <div style="font-size:34px; line-height:1; font-weight:900;">✓</div>
                                <div style="flex:1;">
                                    <div style="font-size:22px; font-weight:800; margin-bottom:4px;">
                                        Successfully uploaded <strong>{len(new)}</strong> rows.
                                    </div>
                                    <div style="font-size:15px; opacity:0.95;">
                                        Go to <span style="font-weight:900;">"Select Employee ID"</span> to view and analyze the new entries.
                                    </div>
                                </div>
                                <div style="flex-shrink:0;">
                                    <a href="#select-employee-id" style="color:white; text-decoration:underline; font-weight:700; font-size:15px;">
                                        Jump to selector →
                                    </a>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:
                # Big red card with actual error message
                st.markdown(
                    f"""
                    <div style="
                        margin-top:18px;
                        padding:22px;
                        border-radius:14px;
                        background-color:#ff4d4f;
                        color:white;
                        font-size:19px;
                        font-weight:700;
                        line-height:1.5;
                    ">
                        ✗ Failed to read uploaded CSV.<br>
                        <div style="font-size:14px; font-weight:400; margin-top:8px; opacity:0.95;">{str(e)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.stop()

    # -----------------------------
    # EMPTY VIEW (placeholder)
    # -----------------------------
    if is_empty:
        st.markdown("### Overview")
        o1,o2,o3 = st.columns(3)
        o1.markdown(small_card("Employee ID", "—"), unsafe_allow_html=True)
        o2.markdown(small_card("Age", "—"), unsafe_allow_html=True)
        o3.markdown(small_card("Position Level", "—"), unsafe_allow_html=True)

        st.markdown("### Key Talent Indexes")
        ki1,ki2,ki3 = st.columns(3)
        ki1.markdown(small_card("Performance Idx", "—"), unsafe_allow_html=True)
        ki2.markdown(small_card("Leadership Idx", "—"), unsafe_allow_html=True)
        ki3.markdown(small_card("Potential Idx", "—"), unsafe_allow_html=True)

        st.markdown("### Character")
        cc1,cc2 = st.columns([0.25,0.75])
        cc1.markdown(small_card("Cluster", "—"), unsafe_allow_html=True)
        cc2.markdown(small_card("Characteristics", "—"), unsafe_allow_html=True)

        st.markdown(build_description_card("—"), unsafe_allow_html=True)
        st.markdown(long_card("HR Recommendations", "—"), unsafe_allow_html=True)
        st.markdown(long_card("Recommended Development Program", "—"), unsafe_allow_html=True)
        st.stop()

    # -----------------------------
    # FEATURE ENGINEERING (safe numeric conversion)
    # -----------------------------
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    for key in ["Leadership_Score","Peer_Review_Score","Performance_Score","Projects_Handled","Training_Hours"]:
        emp[key] = safe_float(emp.get(key, 0))

    emp["Leadership_Index"] = 0.4*emp["Leadership_Score"] + 0.6*emp["Peer_Review_Score"]
    emp["Performance_Index"] = 0.5*emp["Performance_Score"] + 0.2*emp["Projects_Handled"] + 0.3*emp["Peer_Review_Score"]
    emp["Potential_Index"] = 0.4*emp["Training_Hours"] + 0.4*emp["Peer_Review_Score"] + 0.2*emp["Leadership_Score"]

    # For the small single-employee prediction we should scale projects/training using the saved scaler (if available)
    # So we compute scaled values using same scaler used before
    try:
        ph_scaler = joblib.load("cluster_scaler.pkl")
        scaled = ph_scaler.transform(np.array([[emp.get("Projects_Handled",0.0), emp.get("Training_Hours",0.0)]]))
        emp["Projects_Handled_scaled"] = float(scaled[0,0])
        emp["Training_Hours_scaled"] = float(scaled[0,1])
    except Exception:
        # fallback: simple copy (no scaling)
        emp["Projects_Handled_scaled"] = float(emp.get("Projects_Handled",0.0))
        emp["Training_Hours_scaled"] = float(emp.get("Training_Hours",0.0))

    emp["Performance_Consistency"] = emp.get("Performance_Score",0.0) * emp["Projects_Handled_scaled"]
    emp["Growth_Momentum"] = emp["Projects_Handled_scaled"] / (emp["Training_Hours_scaled"] + 1.0)

    # -----------------------------
    # CLUSTERING (robust): try model files; else fallback to centroid-distance using df_ref
    # -----------------------------
    emp["Cluster"] = None
    try:
        if all(k in emp for k in ["Performance_Index","Leadership_Index","Potential_Index"]):
            assigned = None
            # 1) try to use saved scaler + kmeans
            csc_path = "cluster_scaler_3.pkl"
            kpath = "cluster_model.pkl"
            loaded = False
            if os.path.exists(csc_path) and os.path.exists(kpath):
                try:
                    csc = joblib.load(csc_path)
                    kmeans = joblib.load(kpath)
                    arr = csc.transform(np.array([[emp["Performance_Index"], emp["Leadership_Index"], emp["Potential_Index"]]]))
                    lab = kmeans.predict(arr)[0]
                    assigned = int(lab) + 1
                    loaded = True
                except Exception:
                    assigned = None
                    loaded = False

            # 2) fallback: if df_ref contains Cluster and centroids can be computed, use nearest centroid (no file dependency)
            if not loaded:
                if ("Cluster" in df_ref.columns) and (not df_ref.empty) and set(["Performance_Index","Leadership_Index","Potential_Index"]).issubset(df_ref.columns):
                    # compute centroids from master data (grouped by existing Cluster values)
                    try:
                        centroids = (
                            df_ref.groupby("Cluster")[["Performance_Index","Leadership_Index","Potential_Index"]]
                            .mean()
                            .dropna()
                        )
                        if not centroids.empty:
                            # compute distances
                            vec = np.array([emp["Performance_Index"], emp["Leadership_Index"], emp["Potential_Index"]], dtype=float)
                            dists = ((centroids.values - vec.reshape(1, -1))**2).sum(axis=1)
                            idx = int(np.argmin(dists))
                            # centroid index -> get cluster label (index into centroids index)
                            cluster_label = centroids.index[idx]
                            try:
                                assigned = int(cluster_label)
                            except Exception:
                                # if cluster index is not int (e.g. float), coerce
                                assigned = int(float(cluster_label))
                    except Exception:
                        assigned = None

            emp["Cluster"] = assigned
    except Exception:
        emp["Cluster"] = None

    # lookup info safely
    if emp.get("Cluster") is None:
        info = {"Characteristics":"—","Description":"—","HR_Recommendations":"—","HR_Programs":"—"}
    else:
        # ensure we use int key
        try:
            ik = int(emp.get("Cluster"))
        except Exception:
            try:
                ik = int(float(emp.get("Cluster")))
            except Exception:
                ik = None
        info = cluster_map.get(ik, {"Characteristics":"—","Description":"—","HR_Recommendations":"—","HR_Programs":"—"})

    # -----------------------------
    # OVERVIEW (Employee ID, Age, Position Level)
    # -----------------------------
    st.markdown("### Overview")
    oo1,oo2,oo3 = st.columns(3)
    oo1.markdown(small_card("Employee ID", emp.get("Employee_ID","—")), unsafe_allow_html=True)
    oo2.markdown(small_card("Age", emp.get("Age","—")), unsafe_allow_html=True)
    oo3.markdown(small_card("Position Level", emp.get("Current_Position_Level","—")), unsafe_allow_html=True)

    # -----------------------------
    # KEY TALENT INDEXES (3 columns)
    # -----------------------------
    st.markdown("### Key Talent Indexes")
    avg_perf = df_ref["Performance_Index"].mean() if "Performance_Index" in df_ref.columns and not df_ref.empty else 0.0
    avg_lead = df_ref["Leadership_Index"].mean() if "Leadership_Index" in df_ref.columns and not df_ref.empty else 0.0
    avg_pot  = df_ref["Potential_Index"].mean() if "Potential_Index" in df_ref.columns and not df_ref.empty else 0.0

    c1,c2,c3 = st.columns(3)
    c1.markdown(
        small_card(
            "Performance Idx",
            f"{emp.get('Performance_Index',0.0):.2f} | Avg {avg_perf:.2f}",
            "#00bf63" if emp.get("Performance_Index",0.0) >= avg_perf else "#ff5757"
        ),
        unsafe_allow_html=True
    )

    c2.markdown(
        small_card(
            "Leadership Idx",
            f"{emp.get('Leadership_Index',0.0):.2f} | Avg {avg_lead:.2f}",
            "#00bf63" if emp.get("Leadership_Index",0.0) >= avg_lead else "#ff5757"
        ),
        unsafe_allow_html=True
    )

    c3.markdown(
        small_card(
            "Potential Idx",
            f"{emp.get('Potential_Index',0.0):.2f} | Avg {avg_pot:.2f}",
            "#00bf63" if emp.get("Potential_Index",0.0) >= avg_pot else "#ff5757"
        ),
        unsafe_allow_html=True
    )

    # -----------------------------
    # CHARACTER & HR INSIGHTS (proportional)
    # -----------------------------
    st.markdown("### Character")
    cc1,cc2 = st.columns([0.28,0.72])
    cc1.markdown(small_card("Cluster", emp.get("Cluster","—")), unsafe_allow_html=True)
    cc2.markdown(small_card("Characteristics", info.get("Characteristics","—")), unsafe_allow_html=True)

    st.markdown(build_description_card(info.get("Description","—")), unsafe_allow_html=True)
    st.markdown(long_card("HR Recommendations", info.get("HR_Recommendations","—")), unsafe_allow_html=True)
    st.markdown(long_card("Recommended Development Program", info.get("HR_Programs","—")), unsafe_allow_html=True)

    # Save back to globals for compatibility with rest of app (optional)
    globals()["df"] = st.session_state["master_df"]

    # =====================================================================
    # 🧠 Predict Promotion Eligibility — unchanged from original UI
    # =====================================================================
    import math
    import traceback
    import streamlit.components.v1 as components

    st.markdown("## 🧠 Predict Promotion Eligibility")

    # -------------------------
    # Load model
    # -------------------------
    try:
        lr_model = joblib.load("logistic_pipeline.pkl")
    except Exception:
        st.error("Model logistic_pipeline.pkl tidak ditemukan.")
        st.stop()

    # -------------------------
    # Guards
    # -------------------------
    if not emp or not isinstance(emp, (dict, pd.Series)):
        st.info("Isi data talent terlebih dahulu (emp belum tersedia).")
        st.stop()
    if not isinstance(df, pd.DataFrame):
        st.error("Dataset `df` tidak ditemukan atau bukan DataFrame.")
        st.stop()

    def safe_get(d, key, default=0.0):
        try:
            v = d.get(key, default) if isinstance(d, dict) else d.get(key, default)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    input_dict = {
        "Age": safe_get(emp, "Age"),
        "Performance_Index": safe_get(emp, "Performance_Index"),
        "Leadership_Index": safe_get(emp, "Leadership_Index"),
        "Potential_Index": safe_get(emp, "Potential_Index"),
        "Training_Hours": safe_get(emp, "Training_Hours"),
        "Peer_Review_Score": safe_get(emp, "Peer_Review_Score"),
        "Projects_Handled": safe_get(emp, "Projects_Handled"),
        "Performance_Consistency": safe_get(emp, "Performance_Consistency"),
        "Growth_Momentum": safe_get(emp, "Growth_Momentum")
    }
    input_df = pd.DataFrame([input_dict])

    promotion_score = (
        0.30 * input_dict["Performance_Index"] +
        0.25 * input_dict["Potential_Index"] +
        0.20 * input_dict["Leadership_Index"] +
        0.15 * input_dict["Performance_Consistency"] +
        0.10 * input_dict["Growth_Momentum"]
    )

    try:
        df_promo = (
            0.30 * st.session_state["master_df"]["Performance_Index"] +
            0.25 * st.session_state["master_df"]["Potential_Index"] +
            0.20 * st.session_state["master_df"]["Leadership_Index"] +
            0.15 * st.session_state["master_df"]["Performance_Consistency"] +
            0.10 * st.session_state["master_df"]["Growth_Momentum"]
        )
        promo_threshold = df_promo.quantile(0.85)
    except Exception:
        promo_threshold = float("nan")

    components.html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        </style>
        """,
        height=0
    )

    run_prediction = st.button("🔮 Predict Promotion Eligibility", key="predict_promotion_btn_final")
    if not run_prediction:
        st.stop()

    try:
        pred = int(lr_model.predict(input_df)[0])
        is_eligible = (pred == 1)
        status_color = "#00bf63" if is_eligible else "#ff5757"
        promo_str = f"{promotion_score:.2f}"
        threshold_str = f"{promo_threshold:.2f}" if not math.isnan(promo_threshold) else "N/A"

        wrapper_style = "width:calc(100% - 144px); margin-left:72px; margin-right:72px; box-sizing:border-box;"

        big_html = f"""
    <div style="{wrapper_style}">
    <div style="background:#2e307d; border-radius:22px; padding:32px; box-sizing:border-box; font-family:Inter, sans-serif; color:white; overflow:hidden;">
        <h2 style="margin:0; font-size:32px; font-weight:800; color:white;">Promotion Prediction</h2>

        <div style="margin-top:16px; font-size:40px; font-weight:800; color:{status_color};">
        {"Eligible for Promotion" if is_eligible else "Not Eligible"}
        </div>

        <div style="margin-top:12px; font-size:20px; color:white;">
        Promotion Score: <span style="color:{status_color}; font-size:24px; font-weight:700;">{promo_str}</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Threshold (Q85): <span style="color:white; font-size:24px; font-weight:700;">{threshold_str}</span>
        </div>

        <h2 style="margin-top:28px; font-size:28px; font-weight:800; color:white;">Why This Result?</h2>
    """

        features_raw = [
            "Performance_Score","Leadership_Score","Peer_Review_Score",
            "Training_Hours","Projects_Handled","Performance_Consistency","Growth_Momentum"
        ]
        vals = {f: safe_get(emp, f) for f in features_raw}
        means = {}
        for f in features_raw:
            try:
                means[f] = float(st.session_state["master_df"][f].mean())
            except Exception:
                means[f] = float("nan")

        strengths = []
        weaknesses = []
        for f in features_raw:
            v = vals[f]
            avg = means[f]
            if math.isnan(avg):
                continue
            if v >= avg:
                strengths.append((f,v,avg))
            else:
                weaknesses.append((f,v,avg))

        if not is_eligible:
            big_html += '<div style="margin-top:10px; font-size:18px; color:white; line-height:1.7;">'
            if weaknesses:
                big_html += '<div style="font-weight:700; margin-bottom:6px;">Areas Below Expectation:</div>'
                for f,v,avg in weaknesses:
                    big_html += f'<div style="margin-bottom:6px;">{f.replace("_"," ")}: <span style="color:#ff5757; font-weight:700;">{v:.1f}</span> <span style="color:#bbb;">(avg {avg:.1f})</span></div>'
            else:
                big_html += '<div style="margin-bottom:6px;">Semua feature berada di sekitar rata-rata.</div>'
            if strengths:
                big_html += '<div style="margin-top:12px; font-weight:700;">Strengths:</div>'
                for f,v,avg in strengths:
                    big_html += f'<div style="margin-bottom:6px;">{f.replace("_"," ")}: <span style="color:#00bf63; font-weight:700;">{v:.1f}</span> <span style="color:#bbb;">(avg {avg:.1f})</span></div>'
            big_html += '</div>'
        else:
            big_html += '<div style="margin-top:10px; font-size:18px; color:white; line-height:1.7;">'
            if strengths:
                big_html += '<div style="font-weight:700; margin-bottom:6px;">Top Strengths:</div>'
                for f,v,avg in strengths:
                    big_html += f'<div style="margin-bottom:6px;">{f.replace("_"," ")}: <span style="color:#00bf63; font-weight:700;">{v:.1f}</span> <span style="color:#bbb;">(avg {avg:.1f})</span></div>'
            else:
                big_html += '<div style="margin-bottom:6px;">Tidak ada feature menonjol di atas rata-rata.</div>'
            if weaknesses:
                big_html += '<div style="margin-top:12px; font-weight:700;">Can Be Improved:</div>'
                for f,v,avg in weaknesses:
                    big_html += f'<div style="margin-bottom:6px;">{f.replace("_"," ")}: <span style="color:#ff5757; font-weight:700;">{v:.1f}</span> <span style="color:#bbb;">(avg {avg:.1f})</span></div>'
            big_html += '</div>'

        # radar chart (unchanged)
        angles = np.linspace(0, 2*np.pi, len(features_raw), endpoint=False).tolist()
        angles += angles[:1]
        emp_plot = [vals[f] for f in features_raw] + [vals[features_raw[0]]]
        avg_plot = [means[f] if not math.isnan(means[f]) else 0 for f in features_raw] + [means[features_raw[0]] if not math.isnan(means[features_raw[0]]) else 0]

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.plot(angles, avg_plot, color="#bbbbbb", linestyle="dashed", linewidth=2)
        ax.fill(angles, avg_plot, alpha=0.06, color="#bbbbbb")
        ax.plot(angles, emp_plot, color="#00bf63", linewidth=3)
        ax.fill(angles, emp_plot, alpha=0.18, color="#00bf63")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace("_"," ") for f in features_raw], color="white", fontsize=11)
        ax.grid(color="white", alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("white")

        buf = BytesIO()
        plt.savefig(buf, dpi=140, format="png", bbox_inches="tight", pad_inches=0.45, facecolor=fig.get_facecolor())
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        big_html += f'''
        <h2 style="margin-top:28px; font-size:28px; font-weight:800; color:white;">Talent Radar Chart</h2>
        <div style="margin-top:8px;"><img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:12px;" /></div>
        '''

        thr = promo_threshold
        pscore = promotion_score
        if math.isnan(pscore) or math.isnan(thr):
            lvl = "Unknown"; col = "#999999"; dev = ["Insufficient data."]
        else:
            if pscore >= thr:
                lvl = "High Successor Potential"; col = "#00bf63"
                dev = ["Provide advanced leadership exposure.", "Start formal succession mentoring.", "Assess readiness for expanded scope."]
            elif pscore >= thr * 0.9:
                lvl = "Emerging Successor"; col = "#ffaa00"
                dev = ["Start mid-level leadership coaching.", "Increase cross-functional visibility.", "Gradually expand strategic responsibilities."]
            else:
                lvl = "Low Successor Potential"; col = "#ff5757"
                dev = ["Strengthen foundational competencies.", "Improve peer collaboration & influence.", "Enroll in capability-building programs."]

        big_html += f'''
        <h2 style="margin-top:24px; font-size:28px; font-weight:800; color:white;">Succession Potential Indicator</h2>
        <div style="margin-top:8px; font-size:22px; font-weight:800; color:{col};">{lvl}</div>
        <ul style="margin-top:8px; color:white; font-size:17px;">
        '''
        for it in dev:
            big_html += f"<li style='margin-top:6px'>{it}</li>"
        big_html += "</ul>"

        big_html += "</div></div>"

        components.html(big_html, height=1400, scrolling=True)

    except Exception:
        st.error("Error during prediction.")
        st.code(traceback.format_exc())