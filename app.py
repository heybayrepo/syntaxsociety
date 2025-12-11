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

# st.set_page_config(layout="wide")

# === DEFINE CSS FIRST (WAJIB) ===
table_css = """
<style>

/* Background keseluruhan tabel */
[data-testid="stDataFrame"] .st-ag-theme-streamlit-light {
    background-color: #449fe3 !important;
}

/* Cell background */
[data-testid="stDataFrame"] .ag-root-wrapper,
[data-testid="stDataFrame"] .ag-center-cols-container,
[data-testid="stDataFrame"] .ag-cell {
    background-color: #449fe3 !important;
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

/* Light theme */
@media (prefers-color-scheme: light) {
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    [data-testid="stDataFrame"] .ag-root-wrapper,
    [data-testid="stDataFrame"] .ag-cell {
        background-color: #449fe3 !important;
        color: black !important;
    }
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
    [data-testid="stAppViewContainer"] {
        background-color: #2a2b39 !important;
    }
    [data-testid="stDataFrame"] .ag-root-wrapper,
    [data-testid="stDataFrame"] .ag-cell {
        background-color: #449fe3 !important;
        color: white !important;
    }
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# Helper: feature engineering + clustering
# ---------------------------
def apply_feature_engineering_and_clustering(
    df_in,
    cluster_model_path="models/cluster_model.pkl",
    cluster_scaler_path="models/cluster_scaler.pkl",
    cluster_meta_path="models/cluster_metadata.json",
    force_recompute=False
):
    df = df_in.copy()

    required_raw = {
        "Performance_Score", "Leadership_Score", "Training_Hours",
        "Projects_Handled", "Peer_Review_Score"
    }

    if not required_raw.issubset(df.columns):
        return df

    # ==== SAFE TYPE CASTING ====
    for c in required_raw:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ==== INDEX CALCULATIONS ====
    df["Leadership_Index"] = 0.4 * df["Leadership_Score"] + 0.6 * df["Peer_Review_Score"]
    df["Performance_Index"] = (
        0.5 * df["Performance_Score"]
        + 0.2 * df["Projects_Handled"]
        + 0.3 * df["Peer_Review_Score"]
    )
    df["Potential_Index"] = (
        0.4 * df["Training_Hours"]
        + 0.4 * df["Peer_Review_Score"]
        + 0.2 * df["Leadership_Score"]
    )

    # ==== SCALING PROJECTS & TRAINING ====
    scaler_cols = ["Projects_Handled", "Training_Hours"]

    scaler = None
    if os.path.exists(cluster_scaler_path) and not force_recompute:
        try:
            scaler = joblib.load(cluster_scaler_path)
        except:
            scaler = None

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[scaler_cols])
        try: joblib.dump(scaler, cluster_scaler_path)
        except: pass

    scaled = scaler.transform(df[scaler_cols])
    df["Projects_Handled_scaled"] = scaled[:,0]
    df["Training_Hours_scaled"] = scaled[:,1]

    # ==== DERIVED METRICS ====
    df["Performance_Consistency"] = df["Performance_Score"] * df["Projects_Handled_scaled"]
    df["Growth_Momentum"] = df["Projects_Handled_scaled"] / (df["Training_Hours_scaled"] + 1)

    # ==== CLUSTERING FEATURES ====
    cluster_feats = ["Performance_Index","Leadership_Index","Potential_Index"]
    X = df[cluster_feats].values

    scaler3 = None
    scaler3_path = cluster_scaler_path.replace(".pkl","_3.pkl")

    if os.path.exists(scaler3_path) and not force_recompute:
        try:
            scaler3 = joblib.load(scaler3_path)
        except:
            scaler3 = None

    if scaler3 is None:
        scaler3 = StandardScaler()
        scaler3.fit(X)
        try: joblib.dump(scaler3, scaler3_path)
        except: pass

    X_scaled = scaler3.transform(X)

    # ==== KMEANS ====
    kmeans = None
    if os.path.exists(cluster_model_path) and not force_recompute:
        try:
            kmeans = joblib.load(cluster_model_path)
        except:
            kmeans = None

    if kmeans is None:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        try: joblib.dump(kmeans, cluster_model_path)
        except: pass

    df["Cluster"] = kmeans.predict(X_scaled) + 1

    # ============================================================
    # 🔥 FIX: ALWAYS LOAD FULL METADATA, EVEN IF JSON NOT FOUND
    # ============================================================

    default_meta = {
        "Characteristics": {
            "1": "Under Developed With Potential",
            "2": "At-Risk and Underpowered",
            "3": "All Around Top Performer",
            "4": "Consistent Performer or Leader"
        },
        "Description": {
            "1": "Talents in this cluster show strong potential, but their leadership and performance are still developing...",
            "2": "This cluster has the lowest scores across all indices and requires foundational recovery...",
            "3": "This group demonstrates the highest levels of performance, leadership, and potential...",
            "4": "Talents in this cluster excel in execution and reliability; best suited for specialist tracks..."
        },
        "HR_Recommendations": {
            "1": "Provide structured coaching, targeted skill-building, and guided learning exposure.",
            "2": "Rebuild core competencies, strengthen fundamentals, and increase supervision.",
            "3": "Accelerate through leadership programs, strategic rotations, and stretch roles.",
            "4": "Maximize contribution via specialist training, certifications, and expert pathways."
        },
        "HR_Programs": {
            "1": "Performance improvement training, coaching, task ownership.",
            "2": "Core competency recovery, SOP refreshers, regular check-ins.",
            "3": "Leadership bootcamps, rotational assignments, mentorship programs.",
            "4": "Technical certifications, specialist mastery tracks, recognition incentives."
        }
    }

    # TRY LOAD JSON
    if os.path.exists(cluster_meta_path):
        try:
            with open(cluster_meta_path) as f:
                json_meta = json.load(f)
            # merge default with JSON to prevent missing keys
            for section in default_meta:
                json_meta.setdefault(section, default_meta[section])
            cluster_meta = json_meta
        except:
            cluster_meta = default_meta
    else:
        cluster_meta = default_meta
        try:
            with open(cluster_meta_path,"w") as f:
                json.dump(default_meta,f,indent=4)
        except:
            pass

    # ==== APPLY METADATA ====
    df["Characteristics"]       = df["Cluster"].astype(str).map(cluster_meta["Characteristics"])
    df["Description"]           = df["Cluster"].astype(str).map(cluster_meta["Description"])
    df["HR_Recommendations"]    = df["Cluster"].astype(str).map(cluster_meta["HR_Recommendations"])
    df["HR_Programs"]           = df["Cluster"].astype(str).map(cluster_meta["HR_Programs"])

    return df

# ---------------------------
# Load dataset (original path)
# ---------------------------
DATA_PATH = "data/Clean/dataset_for_dashboard.csv"
FALLBACK_RAW_PATH = "data/Clean/dataset_for_dashboard.csv"

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

# >>> TARUH CSS DI SINI <<<
st.markdown("""
    <style>    
        /* Perbesar font semua tab */
        .stTabs [role="tab"] p {
            font-size: 16px !important;
        }
    
        /* Garis bawah tab aktif */
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #00bf63 !important;
        }

        /* Warna teks tab aktif */
        .stTabs [aria-selected="true"] p {
            color: #00bf63 !important;
            font-weight: 700 !important;
        }

        /* Warna teks saat hover */
        .stTabs [role="tab"]:hover p {
            color: #00bf63 !important;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# CREATE TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Talent Overview", "Talent Performance", "Talent Predictor"])

# ======================================================================
# TAB 1 — TALENT OVERVIEW
# ======================================================================

with tab1:
    st.markdown("""
    <style>

    /* Light theme */
    @media (prefers-color-scheme: light) {
        .talent-desc {
            color: #000000 !important;   /* hitam */
        }
    }

    /* Dark theme */
    @media (prefers-color-scheme: dark) {
        .talent-desc {
            color: #d0d0d0 !important;   /* abu terang */
        }
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📸 Talent Overview")

    st.markdown(
        "<p class='talent-desc' style='font-size:16px; margin-top:-10px;'>"
        "This section provides a clear view of our talent composition across roles, ages, and performance groups. "
        "The insights help HR identify where strengths are concentrated and where additional support may be needed, "
        "including understanding the monthly cost associated with low-performing talent."
        "</p>",
        unsafe_allow_html=True
    )

    colA, colB = st.columns(2)

    def overview_card(title, value):
        return f"""
            <div style="
                border: 3px solid #449fe3;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 15px;
                background-color: #449fe3;
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
            <div style="font-size:22px;font-weight:600;color:ffffff;">{r['Current_Position_Level']}</div>
            <div style="font-size:22px;color:#ffffff;">{int(r['Total_Talent'])} talent</div>
        </div>
        """

    st.markdown(
        f"""
        <div style="width:100%;border:3px solid #449fe3;border-radius:12px;
                    padding:20px;margin-top:10px;background-color:#449fe3;">
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
        fig.patch.set_facecolor("#449fe3")
        ax.set_facecolor("#449fe3")

        st.markdown(
            f"""
            <div style="border:3px solid #449fe3;border-radius:12px;
                        padding:20px;margin-top:20px;background-color:#449fe3;">
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
        fig2.patch.set_facecolor("#449fe3")
        ax2.set_facecolor("#449fe3")

        st.markdown(
            f"""
            <div style="border:3px solid #449fe3;border-radius:12px;
                        padding:20px;margin-top:20px;background-color:#449fe3;">
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
        <div style='font-size:22px; color:#ffffff;'>{cluster_avg}</div>
    </div>
    """

    cluster_card = f"""
    <div style='width:100%; border:3px solid #449fe3; border-radius:12px;
                padding:20px; margin-top:15px; background-color:#449fe3;'>
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
            "<div style='width:100%; border:3px solid #449fe3; border-radius:12px;"
            " padding:20px; margin-top:15px; background-color:#449fe3;"
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
            formatted_salary = "Rp" + "{:,.0f}".format(total_salary).replace(",", ".")
        except:
            formatted_salary = "Rp" + str(total_salary)

        html_card = (
            "<div style='width:100%; border:3px solid #449fe3; border-radius:12px;"
            " padding:20px; margin-top:15px; background-color:#449fe3;"
            " font-family:Inter, sans-serif;'>"

            "<div style='font-size:22px; font-weight:700; margin-bottom:12px; color:white;'>"
            "Potential Loss"
            "</div>"

            "<div style='display:flex; justify-content:space-between; align-items:center; padding:8px 0;'>"
            "<div style='font-size:16px; color:#ffffff;'>"
            "Total monthly salary paid to low-performing talent"
            "</div>"
            f"<div style='font-size:28px; font-weight:700; color:#ff5757;'>{formatted_salary}</div>"
            "</div>"

            "<div style='font-size:13px; margin-top:10px; color:#ffffff;'>"
            "This represents the estimated monthly salary load associated with low-performing talent, "
            "a direct indicator of potential productivity loss from a business perspective."
            "</div>"

            "</div>"
        )

        st.markdown(html_card, unsafe_allow_html=True)


# ======================================================================
# TAB 2 — TALENT PERFORMANCE (ASLI)
# ======================================================================
with tab2:

    # ===========================================================
    # 🌱 Promotion-Ready Talent — compact + scrollable table
    # ===========================================================
    st.markdown("## 🌱 Promotion-Ready")

    st.markdown(
    "<p style='color:#d0d0d0; font-size:16px; margin-top:-10px;'>"
    "These employees have met the required criteria for promotion, showing strong readiness to step into greater responsibility."
    "</p>",
    unsafe_allow_html=True
    )

    # attempt load model (silence warning handled below)
    try:
        lr_model = joblib.load("models/logistic_pipeline.pkl")
    except Exception:
        lr_model = None

    if lr_model is None:
        st.info("Promotion model tidak ditemukan. Letakkan `logistic_pipeline.pkl` di folder /models untuk mengaktifkan.")
    else:
        df_temp = st.session_state.get("master_df", pd.DataFrame()).copy()
        if df_temp.empty:
            st.info("Dataset kosong — tidak ada kandidat untuk dianalisis.")
        else:
            # pastikan semua feature ada — fallback numeric 0 agar model tidak crash
            features = [
                "Age","Performance_Index","Leadership_Index","Potential_Index",
                "Training_Hours","Peer_Review_Score","Projects_Handled",
                "Performance_Consistency","Growth_Momentum"
            ]
            for c in features:
                if c not in df_temp.columns:
                    df_temp[c] = 0.0

            # compute promotion score (sama formula di Tab 3)
            df_temp["Promotion_Score"] = (
                0.30 * df_temp["Performance_Index"] +
                0.25 * df_temp["Potential_Index"] +
                0.20 * df_temp["Leadership_Index"] +
                0.15 * df_temp["Performance_Consistency"] +
                0.10 * df_temp["Growth_Momentum"]
            )

            # Predict eligibility (if model fails, fallback to Promotion_Score threshold logic)
            try:
                df_temp["Eligible_Pred"] = lr_model.predict(df_temp[features])
            except Exception:
                # fallback: mark top percentile as eligible (best-effort)
                thr = df_temp["Promotion_Score"].quantile(0.90)
                df_temp["Eligible_Pred"] = (df_temp["Promotion_Score"] >= thr).astype(int)
                st.warning("Model prediksi gagal digunakan; fallback ke threshold Promotion_Score (Q90).")

            # filter eligible
            eligible_df = df_temp[df_temp["Eligible_Pred"] == 1].copy()

            if eligible_df.empty:
                st.info("Tidak ada kandidat eligible saat ini menurut model/fallback.")
            else:
                # sort by Leadership_Index (desc) as requested and keep 10 rows
                top_eligible = eligible_df.sort_values("Leadership_Index", ascending=False).head(10)

                # columns to display (rename to friendly names if mau)
                display_cols = ["Employee_ID", "Age", "Current_Position_Level", "Leadership_Index", "Performance_Index"]
                # ensure cols exist
                for col in display_cols:
                    if col not in top_eligible.columns:
                        top_eligible[col] = "—"

                st.markdown("**These talents are eligible for promotion**")
                st.dataframe(
                    top_eligible[display_cols].reset_index(drop=True),
                    hide_index=True,
                    use_container_width=True,
                    height=360  # fixed height -> will show scrollbar if rows exceed visible area
                )

    # ⭐ Top Talent
    st.markdown("## ⭐ Top Talent")

    st.markdown(
    "<p style='color:#d0d0d0; font-size:16px; margin-top:-10px;'>"
    "These high-performing employees show strong potential for future advancement, even if they have not yet met all promotion requirements."
    "</p>",
    unsafe_allow_html=True
    )


    colA, colB = st.columns(2)

    with colA:
        category = st.selectbox(
            'Select ranking category:',
            ['Best Performing', 'Best Leadership', 'Best Potential'],
            key="ranking_category"
        )

    with colB:
        position_options = (
            ['All Levels'] +
            sorted(st.session_state["master_df"]['Current_Position_Level'].dropna().unique().tolist())
            if not st.session_state["master_df"].empty else ['All Levels']
        )
        selected_level = st.selectbox(
            'Filter by Position Level:',
            position_options,
            key="position_filter"
        )

    df_filtered = (
        st.session_state["master_df"]
        if selected_level == 'All Levels'
        else st.session_state["master_df"][st.session_state["master_df"]['Current_Position_Level'] == selected_level]
    )

    # ================================
    # BEST PERFORMING
    # ================================
    if category == 'Best Performing':

        ranked = (
            df_filtered.sort_values('Performance_Index', ascending=False).head(10)
            if 'Performance_Index' in df_filtered.columns else pd.DataFrame()
        )

        cols = [
            'Employee_ID',
            'Current_Position_Level',
            'Performance_Index',
            'Performance_Score',
            'Cluster'
        ]

        ranked = ranked[cols] if not ranked.empty else ranked

        st.dataframe(ranked, use_container_width=True, hide_index=True)


    # ================================
    # BEST LEADERSHIP
    # ================================
    elif category == 'Best Leadership':

        ranked = (
            df_filtered.sort_values('Leadership_Index', ascending=False).head(10)
            if 'Leadership_Index' in df_filtered.columns else pd.DataFrame()
        )

        cols = [
            'Employee_ID',
            'Current_Position_Level',
            'Leadership_Index',
            'Leadership_Score',
            'Cluster'
        ]

        ranked = ranked[cols] if not ranked.empty else ranked

        st.dataframe(ranked, use_container_width=True, hide_index=True)


    # ================================
    # BEST POTENTIAL (tetap seperti sebelumnya)
    # ================================
    else:
        ranked = (
            df_filtered.sort_values('Potential_Index', ascending=False).head(10)
            if 'Potential_Index' in df_filtered.columns else pd.DataFrame()
        )
        cols = [
            'Employee_ID',
            'Current_Position_Level',
            'Potential_Index',
            'Growth_Momentum',
            'Training_Hours'
        ]
        ranked = ranked[cols] if not ranked.empty else ranked

        st.dataframe(ranked, use_container_width=True, hide_index=True)


    # ===========================================================
    # ⚠️ HIGH RISK TALENT
    # ===========================================================
    st.markdown("## ⚠️ High Risk Talent")

    st.markdown(
    "<p style='color:#d0d0d0; font-size:16px; margin-top:-10px;'>"
    "These employees fall into lower performance ranges and may need targeted development, mentoring, or closer support."
    "</p>",
    unsafe_allow_html=True
    )
    colR1, colR2 = st.columns(2)

    with colR1:
        risk_category = st.selectbox(
            'Select ranking category:',
            ['Low Performing', 'Low Leadership', 'Low Potential'],
            key="risk_category"
        )

    with colR2:
        position_options_risk = position_options
        selected_level_risk = st.selectbox(
            'Filter by Position Level:',
            position_options_risk,
            key="risk_position_filter"
        )

    df_risk = (
        st.session_state["master_df"]
        if selected_level_risk == 'All Levels'
        else st.session_state["master_df"][st.session_state["master_df"]['Current_Position_Level'] == selected_level_risk]
    )


    # ================================
    # LOW PERFORMING (mirror Best Performing)
    # ================================
    if risk_category == 'Low Performing':

        ranked = (
            df_risk.sort_values('Performance_Index', ascending=True).head(10)
            if 'Performance_Index' in df_risk.columns else pd.DataFrame()
        )

        cols = [
            'Employee_ID',
            'Current_Position_Level',
            'Performance_Index',
            'Performance_Score',
            'Cluster'
        ]

        ranked = ranked[cols] if not ranked.empty else ranked

        st.dataframe(ranked, use_container_width=True, hide_index=True)


    # ================================
    # LOW LEADERSHIP (mirror Best Leadership)
    # ================================
    elif risk_category == 'Low Leadership':

        ranked = (
            df_risk.sort_values('Leadership_Index', ascending=True).head(10)
            if 'Leadership_Index' in df_risk.columns else pd.DataFrame()
        )

        cols = [
            'Employee_ID',
            'Current_Position_Level',
            'Leadership_Index',
            'Leadership_Score',
            'Cluster'
        ]

        ranked = ranked[cols] if not ranked.empty else ranked

        st.dataframe(ranked, use_container_width=True, hide_index=True)


    # ================================
    # LOW POTENTIAL (tetap)
    # ================================
    else:
        ranked = (
            df_risk.sort_values('Potential_Index', ascending=True).head(10)
            if 'Potential_Index' in df_risk.columns else pd.DataFrame()
        )

        cols = [
            'Employee_ID',
            'Current_Position_Level',
            'Potential_Index',
            'Growth_Momentum',
            'Training_Hours'
        ]

        ranked = ranked[cols] if not ranked.empty else ranked

        st.dataframe(ranked, use_container_width=True, hide_index=True)


# ===== TAB 3 — PART 1/3 =====
# (Paste this where your original `with tab3:` block should start)

with tab3:

    df_ref = st.session_state.get("master_df", pd.DataFrame())

    # ===========================================================
    # CARD BUILDERS (no design changes)
    # ===========================================================
    def small_card(title, value, color="white", bg="#449fe3"):
        return f"""
        <div style='border:3px solid #2e307d;border-radius:12px;
                    padding:16px;margin-bottom:15px;background-color:{bg};'>
            <h3 style='color:white;font-size:20px;margin:0'>{title}</h3>
            <div style='margin-top:12px;font-size:22px;color:{color}'>{value}</div>
        </div>
        """

    def build_description_card(text):
        # ensure long text wraps properly
        safe_text = str(text) if text is not None else "—"
        return f"""
        <div style='border:3px solid #2e307d;border-radius:12px;
                    padding:20px;margin-top:20px;background-color:#449fe3;'>
            <div style='font-size:22px;font-weight:700;margin-bottom:12px;color:white'>Description</div>
            <div style='font-size:18px;color:#ddd; white-space:normal; word-wrap:break-word;'>{safe_text}</div>
        </div>
        """

    def long_card(title, content):
        safe_content = str(content) if content is not None else "—"
        return f"""
        <div style='border:3px solid #2e307d;border-radius:12px;
                    padding:20px;margin-top:20px;background-color:#00bf63;'>
            <div style='font-size:22px;font-weight:700;color:white;margin-bottom:12px'>{title}</div>
            <div style='font-size:18px;color:white; white-space:normal; word-wrap:break-word;'>{safe_content}</div>
        </div>
        """

    st.markdown("## 🔎 Talent Predictor")

    st.markdown(
    "<p style='color:#d0d0d0; font-size:16px; margin-top:-10px;'>"
    "This tool allows HR to review individual employee profiles, understand their talent cluster and characteristics, "
    "and assess promotion eligibility using the prediction model. It supports data-driven decisions by combining "
    "employee attributes with machine learning algorithms."
    "</p>",
    unsafe_allow_html=True
    )

    st.markdown("### Select Talent Input Method")

    # ===========================================================
    # METADATA MAP (build from master_df if present)
    # ===========================================================
    cluster_map = {}
    if {"Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"}.issubset(df_ref.columns):
        tmp = (
            df_ref[["Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"]]
            .drop_duplicates("Cluster")
            .set_index("Cluster")
            .to_dict("index")
        )
        for k,v in tmp.items():
            try:
                cluster_map[int(k)] = v
            except Exception:
                # try float->int
                try:
                    cluster_map[int(float(k))] = v
                except Exception:
                    continue

    # ===========================================================
    # SELECT INPUT METHOD
    # ===========================================================
    mode = st.radio(
        "",
        [
            "Select employee ID",
            "Predict employee cluster and characteristics",
            "Upload employee data in bulk using CSV"
        ],
        horizontal=True
    )

    # default placeholders
    emp = {}
    is_empty = False

    # ===========================================================
    # 1) SELECT EMPLOYEE ID
    # ===========================================================
    if mode == "Select employee ID":
        st.markdown("#### Select Current Employee")
        col1, col2 = st.columns(2)

        with col1:
            ids = df_ref["Employee_ID"].astype(str).unique().tolist() if not df_ref.empty else []
            emp_id_sel = st.selectbox("Select Employee ID:", ["None"] + ids, key="tp_sel")

        with col2:
            typed = st.text_input("Or type Employee ID:", key="tp_typed")

        emp_id = typed.strip() if typed.strip() and (typed.strip() in ids) else emp_id_sel

        if emp_id == "None" or emp_id not in ids:
            is_empty = True
            emp = {}
        else:
            # safe extraction (use iloc[0] if exists)
            row = df_ref[df_ref["Employee_ID"].astype(str) == emp_id]
            if row.empty:
                is_empty = True
                emp = {}
            else:
                emp = row.iloc[0].to_dict()

    # ===========================================================
    # 2) MANUAL INPUT FORM
    # ===========================================================
    elif mode == "Predict employee cluster and characteristics":
        st.markdown("### Predict Employee Cluster and Characteristics")

        emp_id_input = st.text_input("Employee ID (optional):")

        def to_int_or_none(v):
            try:
                if v is None:
                    return None
                v2 = str(v).strip()
                if v2 == "" or v2.lower() == "choose":
                    return None
                return int(v2)
            except Exception:
                return None

        age = to_int_or_none(st.text_input("Age", placeholder="18-70"))
        perf = to_int_or_none(st.selectbox("Performance Score (1–5)", ["Choose"] + list(range(1,6))))
        lead = to_int_or_none(st.selectbox("Leadership Score (0–100)", ["Choose"] + list(range(0,101,10))))
        train = to_int_or_none(st.selectbox("Training Hours (0–200)", ["Choose"] + list(range(0,201,20))))
        proj = to_int_or_none(st.selectbox("Projects Handled (0–20)", ["Choose"] + list(range(0,21))))
        peer = to_int_or_none(st.selectbox("Peer Review Score (0–100)", ["Choose"] + list(range(0,101,10))))
        level = st.selectbox("Current Position Level", ["Choose","Junior","Mid","Senior","Lead"])

        required = (age is not None and perf is not None and lead is not None and train is not None and
                    proj is not None and peer is not None and level != "Choose")

        if not required:
            emp = {}
            is_empty = True
        else:
            emp = {
                "Employee_ID": emp_id_input.strip() if emp_id_input and emp_id_input.strip() else "—",
                "Age": age,
                "Performance_Score": perf,
                "Leadership_Score": lead,
                "Training_Hours": train,
                "Projects_Handled": proj,
                "Peer_Review_Score": peer,
                "Current_Position_Level": level
            }
            is_empty = False

    # ===========================================================
    # 3) UPLOAD CSV  (FULL FIXED BLOCK)
    # ===========================================================
    elif mode == "Upload employee data in bulk using CSV":
        st.markdown("Upload CSV")

        template = pd.DataFrame({
            "Employee_ID": ["EMP0001"],
            "Age": [30],
            "Performance_Score": [5],
            "Leadership_Score": [60],
            "Training_Hours": [40],
            "Projects_Handled": [5],
            "Peer_Review_Score": [75],
            "Current_Position_Level": ["Senior"]
        })

        st.download_button("Download template CSV", template.to_csv(index=False), "template.csv")

        uploaded = st.file_uploader("Upload CSV:", type=["csv"])

        if uploaded:
            try:
                new = pd.read_csv(uploaded)

                required_cols = set(template.columns)
                if not required_cols.issubset(new.columns):
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
                            ✗ Missing required columns in uploaded CSV.<br>
                            Please use the provided template.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.stop()

                # PROCESS: feature engineering & clustering for uploaded rows
                enriched = apply_feature_engineering_and_clustering(new.copy())

                # APPEND to master dataframe in session state
                st.session_state["master_df"] = pd.concat([df_ref, enriched], ignore_index=True)

                # ===========================================================
                # SUCCESS CARD
                # ===========================================================
                st.markdown(
                    f"""
                    <div style="
                        margin-top:18px;
                        padding:22px;
                        border-radius:14px;
                        background-color:#00bf63;
                        color:white;
                        font-size:19px;
                        font-weight:700;
                        line-height:1.5;
                    ">
                        ✓ Uploaded {len(new)} rows successfully.<br>
                        Go to <strong>'Select employee ID'</strong> to view/inspect.
                    </div>
                    """, unsafe_allow_html=True
                )

                # === LOAD MODEL ===
                model_path = "models/logistic_pipeline.pkl"
                if "elig_model" not in st.session_state:
                    st.session_state["elig_model"] = joblib.load(model_path)
                model = st.session_state["elig_model"]

                # === FEATURES USED IN TRAINING ===
                logreg_features = [
                    "Age",
                    "Performance_Index",
                    "Leadership_Index",
                    "Potential_Index",
                    "Training_Hours",
                    "Peer_Review_Score",
                    "Projects_Handled",
                    "Performance_Consistency",
                    "Growth_Momentum"
                ]

                # === PREDICT ELIGIBILITY ===
                X_new = enriched[logreg_features]
                enriched["Eligible_New"] = model.predict(X_new).astype(int)

                # === 📌 CARD ELIGIBLE + LIST EMPLOYEE ID (TARUH DI SINI) ===
                eligible_ids = enriched.loc[enriched["Eligible_New"] == 1, "Employee_ID"].tolist()
                eligible_count = len(eligible_ids)
                eligible_list_str = ", ".join(eligible_ids) if eligible_ids else "-"

                st.markdown(
                    f"""
                    <div style="
                        margin-top:18px;
                        padding:18px;
                        border-radius:12px;
                        background-color:#1e90ff;
                        color:white;
                        font-size:18px;
                        font-weight:600;
                    ">
                        {eligible_count} employees in this uploaded file are
                        <strong>predicted to be promotion eligible</strong>.
                        <br><br>
                        <span style="
                            font-size:15px;
                            font-weight:400;
                            line-height:1.6;
                        ">
                            <strong>Employee IDs:</strong> {eligible_list_str}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # === SHOW PROCESSED + CLUSTERED DATA ===
                st.markdown("### Preview of your processed and clustered data")
                st.dataframe(enriched)

            except Exception as e:
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
                        <span style="font-size:16px; font-weight:400;">{e}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.stop()

    # ===========================================================
    # EMPTY VIEW (placeholder)
    # ===========================================================
    if is_empty:
        st.markdown("### Overview")
        a,b,c = st.columns(3)
        a.markdown(small_card("Employee ID","—"), unsafe_allow_html=True)
        b.markdown(small_card("Age","—"), unsafe_allow_html=True)
        c.markdown(small_card("Position Level","—"), unsafe_allow_html=True)

        st.markdown("### Key Talent Indexes")
        x,y,z = st.columns(3)
        x.markdown(small_card("Performance Idx","—"), unsafe_allow_html=True)
        y.markdown(small_card("Leadership Idx","—"), unsafe_allow_html=True)
        z.markdown(small_card("Potential Idx","—"), unsafe_allow_html=True)

        st.markdown("### Character")
        cc1,cc2 = st.columns([0.28,0.72])
        cc1.markdown(small_card("Cluster","—"), unsafe_allow_html=True)
        cc2.markdown(small_card("Characteristics","—"), unsafe_allow_html=True)

        st.markdown(build_description_card("—"), unsafe_allow_html=True)
        st.markdown(long_card("HR Recommendations","—"), unsafe_allow_html=True)
        st.markdown(long_card("Recommended Development Program","—"), unsafe_allow_html=True)
        st.stop()

# ===== end of PART 1/3 =====

# ===== TAB 3 — PART 2/3 =====
# (Paste this immediately after Part 1)

    # ===========================================================
    # FEATURE ENGINEERING (manual or selected employee)
    # ===========================================================
    def safe_float(v):
        try: 
            return float(v)
        except:
            return 0.0

    for k in ["Leadership_Score","Peer_Review_Score","Performance_Score",
              "Projects_Handled","Training_Hours"]:
        emp[k] = safe_float(emp.get(k, 0))

    emp["Leadership_Index"] = (
        0.4 * emp["Leadership_Score"] + 
        0.6 * emp["Peer_Review_Score"]
    )

    emp["Performance_Index"] = (
        0.5 * emp["Performance_Score"] +
        0.2 * emp["Projects_Handled"] +
        0.3 * emp["Peer_Review_Score"]
    )

    emp["Potential_Index"] = (
        0.4 * emp["Training_Hours"] +
        0.4 * emp["Peer_Review_Score"] +
        0.2 * emp["Leadership_Score"]
    )

    # ===========================================================
    # STABLE CLUSTER ASSIGNMENT (tidak berubah-ubah!)
    # ===========================================================
    emp["Cluster"] = None
    try:
        scaler_path = "models/cluster_scaler_3.pkl"
        model_path  = "models/cluster_model.pkl"

        if os.path.exists(scaler_path) and os.path.exists(model_path):

            scaler = joblib.load(scaler_path)
            kmeans = joblib.load(model_path)

            arr = scaler.transform([[
                emp["Performance_Index"],
                emp["Leadership_Index"],
                emp["Potential_Index"]
            ]])

            raw_label = int(kmeans.predict(arr)[0])  # 0–3

            # hitung centroid untuk ranking stabil
            cent = pd.DataFrame(
                kmeans.cluster_centers_,
                columns=["Performance_Index","Leadership_Index","Potential_Index"]
            )
            cent["score"] = cent.sum(axis=1)

            # ranking cluster dari terbaik → terburuk
            ordered = cent.sort_values("score", ascending=False).index.tolist()

            # MAPPING FINAL (fix, tidak akan berubah)
            stable_map = {
                ordered[0]: 3,   # High Performer
                ordered[1]: 4,   # Consistent Performer
                ordered[2]: 1,   # Underdeveloped Potential
                ordered[3]: 2    # At-Risk & Underpowered
            }

            emp["Cluster"] = stable_map.get(raw_label, 1)

        else:
            # fallback paling aman
            emp["Cluster"] = 1

    except Exception:
        emp["Cluster"] = 1

    # ===========================================================
    # METADATA LOOKUP (Characteristics, Description, HR Programs)
    # ===========================================================
    info = cluster_map.get(
        int(emp["Cluster"]),
        {
            "Characteristics": "—",
            "Description": "—",
            "HR_Recommendations": "—",
            "HR_Programs": "—"
        }
    )

    # ===========================================================
    # OVERVIEW SECTION
    # ===========================================================
    st.markdown("### Overview")
    oo1, oo2, oo3 = st.columns(3)

    oo1.markdown(small_card("Employee ID", emp.get("Employee_ID", "—")),
                 unsafe_allow_html=True)
    oo2.markdown(small_card("Age", emp.get("Age", "—")),
                 unsafe_allow_html=True)
    oo3.markdown(small_card("Position Level",
                            emp.get("Current_Position_Level", "—")),
                 unsafe_allow_html=True)

    # ===========================================================
    # KEY TALENT INDEXES
    # ===========================================================
    st.markdown("### Key Talent Indexes")
    k1, k2, k3 = st.columns(3)

    k1.markdown(small_card("Performance Idx",
                           f"{emp['Performance_Index']:.2f}"),
                unsafe_allow_html=True)

    k2.markdown(small_card("Leadership Idx",
                           f"{emp['Leadership_Index']:.2f}"),
                unsafe_allow_html=True)

    k3.markdown(small_card("Potential Idx",
                           f"{emp['Potential_Index']:.2f}"),
                unsafe_allow_html=True)

    # ===========================================================
    # CHARACTER + HR INSIGHTS
    # ===========================================================
    st.markdown("### Character")
    cc1, cc2 = st.columns([0.28, 0.72])

    cc1.markdown(
        small_card("Cluster", emp["Cluster"]),
        unsafe_allow_html=True
    )
    cc2.markdown(
        small_card("Characteristics", info["Characteristics"]),
        unsafe_allow_html=True
    )

    st.markdown(
        build_description_card(info["Description"]),
        unsafe_allow_html=True
    )
    st.markdown(
        long_card("HR Recommendations", info["HR_Recommendations"]),
        unsafe_allow_html=True
    )
    st.markdown(
        long_card("Recommended Development Program", info["HR_Programs"]),
        unsafe_allow_html=True
    )

# ===== end of PART 2/3 =====

# ===== TAB 3 — PART 3/3: Prediction + Radar + Explanations =====
# (Paste this immediately after Part 2)

    # ===========================================================
    # PREDICTION ENGINE (FULL — radar, strengths/weaknesses, succession)
    # ===========================================================
    import math
    import traceback
    import streamlit.components.v1 as components

    # load model (try common paths)
    lr_model = None
    for p in ["models/logistic_pipeline.pkl", "logistic_pipeline.pkl", "models/logistic_pipeline.joblib"]:
        try:
            if os.path.exists(p):
                lr_model = joblib.load(p)
                break
        except Exception:
            lr_model = None

    if lr_model is None:
        st.error("Model `models/logistic_pipeline.pkl` tidak ditemukan. Letakkan model di folder /models.")
        st.stop()

    def safe_get(d, key):
        try:
            v = d.get(key, 0) if isinstance(d, dict) else d[key]
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    # build input dict used by model
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
        df_promo_series = (
            0.30 * df_ref["Performance_Index"] +
            0.25 * df_ref["Potential_Index"] +
            0.20 * df_ref["Leadership_Index"] +
            0.15 * df_ref["Performance_Consistency"] +
            0.10 * df_ref["Growth_Momentum"]
        )
        promo_threshold = df_promo_series.quantile(0.85)
    except Exception:
        promo_threshold = float("nan")

    # Button (if Part2 already had a button, duplicating is safe here as user will paste sequentially)
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)  # spacer
    run_prediction = st.button("🔮 Predict Promotion Eligibility (Run)")

    if not run_prediction:
        st.stop()

    # perform prediction & render full analysis
    try:
        pred_raw = lr_model.predict(input_df)[0]
        pred = int(pred_raw)
        is_eligible = (pred == 1)
        status_color = "#00bf63" if is_eligible else "#ff5757"
        promo_str = f"{promotion_score:.2f}"
        threshold_str = f"{promo_threshold:.2f}" if not math.isnan(promo_threshold) else "N/A"

        # ensure Inter font loads for consistent look
        components.html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        </style>
        """, height=0)

        wrapper_style = "width:calc(100% - 144px); margin-left:72px; margin-right:72px; box-sizing:border-box;"

        big_html = f"""
        <div style="{wrapper_style}">
        <div style="background:#2e307d; border-radius:22px; padding:32px; box-sizing:border-box;
                    font-family:Inter, sans-serif; color:white; overflow:hidden;">
            <h2 style="margin:0; font-size:32px; font-weight:800; color:white;">Promotion Prediction</h2>

            <div style="margin-top:16px; font-size:40px; font-weight:800; color:{status_color};">
                {"Eligible for Promotion" if is_eligible else "Not Eligible"}
            </div>

            <div style="margin-top:12px; font-size:20px; color:white;">
                Promotion Score:
                <span style="color:{status_color}; font-size:24px; font-weight:700;">{promo_str}</span>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Threshold (Q85):
                <span style="color:white; font-size:24px; font-weight:700;">{threshold_str}</span>
            </div>

            <h2 style="margin-top:28px; font-size:28px; font-weight:800; color:white;">Why This Result?</h2>
        """

        # feature-level analysis
        features_raw = [
            "Performance_Score","Leadership_Score","Peer_Review_Score",
            "Training_Hours","Projects_Handled","Performance_Consistency","Growth_Momentum"
        ]

        vals = {f: safe_get(emp, f) for f in features_raw}
        means = {}
        for f in features_raw:
            try:
                means[f] = float(df_ref[f].mean())
            except Exception:
                means[f] = float("nan")

        strengths = []
        weaknesses = []
        for f in features_raw:
            v = vals[f]
            avg = means[f]
            if math.isnan(avg):
                # skip comparison if no population mean
                continue
            if v >= avg:
                strengths.append((f, v, avg))
            else:
                weaknesses.append((f, v, avg))

        # Always show both sections if data exist:
        # If eligible: show Top Strengths and then "Can Be Improved" if any weaknesses
        # If not eligible: show Areas Below Expectation and then "Strengths" if any
        big_html += '<div style="margin-top:10px; font-size:18px; line-height:1.7;">'
        if is_eligible:
            big_html += '<div style="font-weight:700; margin-bottom:6px; color:#00bf63;">Top Strengths:</div>'
            if strengths:
                for f,v,avg in strengths:
                    big_html += f"<div>{f.replace('_',' ')}: <span style='color:#00bf63;font-weight:700'>{v:.1f}</span> <span style='color:#bbb'>(avg {avg:.1f})</span></div>"
            else:
                big_html += "<div>No feature clearly above population mean.</div>"

            if weaknesses:
                big_html += '<div style="margin-top:12px; font-weight:700; color:#ffaa00;">Can Be Improved:</div>'
                for f,v,avg in weaknesses:
                    big_html += f"<div>{f.replace('_',' ')}: <span style='color:#ff5757;font-weight:700'>{v:.1f}</span> <span style='color:#bbb'>(avg {avg:.1f})</span></div>"
        else:
            big_html += '<div style="font-weight:700; margin-bottom:6px; color:#ff5757;">Areas Below Expectation:</div>'
            if weaknesses:
                for f,v,avg in weaknesses:
                    big_html += f"<div>{f.replace('_',' ')}: <span style='color:#ff5757;font-weight:700'>{v:.1f}</span> <span style='color:#bbb'>(avg {avg:.1f})</span></div>"
            else:
                big_html += "<div>All features are around or above the mean.</div>"

            if strengths:
                big_html += '<div style="margin-top:12px; font-weight:700; color:#00bf63;">Strengths:</div>'
                for f,v,avg in strengths:
                    big_html += f"<div>{f.replace('_',' ')}: <span style='color:#00bf63;font-weight:700'>{v:.1f}</span> <span style='color:#bbb'>(avg {avg:.1f})</span></div>"
        big_html += '</div>'

        # -------------------------
        # RADAR CHART (matplotlib → base64)
        # -------------------------
        angles = np.linspace(0, 2*np.pi, len(features_raw), endpoint=False).tolist()
        angles += angles[:1]

        emp_plot = [vals[f] if not math.isnan(vals[f]) else 0 for f in features_raw] + [vals[features_raw[0]] if not math.isnan(vals[features_raw[0]]) else 0]
        avg_plot = [means[f] if not math.isnan(means[f]) else 0 for f in features_raw] + [means[features_raw[0]] if not math.isnan(means[features_raw[0]]) else 0]

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # plot population avg and employee
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
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0.45, facecolor=fig.get_facecolor())
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        big_html += f"""
        <h2 style="margin-top:28px; font-size:28px; font-weight:800;">Talent Radar Chart</h2>
        <div style="margin-top:8px;"><img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:12px;" /></div>
        """

        # -------------------------
        # SUCCESSION POTENTIAL / RECOMMENDATIONS
        # -------------------------
        thr = promo_threshold
        ps  = promotion_score

        if math.isnan(ps) or math.isnan(thr):
            lvl = "Unknown"; col = "#999999"; dev = ["Insufficient data."]
        else:
            if ps >= thr:
                lvl = "High Successor Potential"; col = "#00bf63"
                dev = [
                    "Provide advanced leadership exposure.",
                    "Start formal succession mentoring.",
                    "Assess readiness for expanded scope."
                ]
            elif ps >= thr * 0.9:
                lvl = "Emerging Successor"; col = "#ffaa00"
                dev = [
                    "Start mid-level leadership coaching.",
                    "Increase cross-functional visibility.",
                    "Gradually expand strategic responsibilities."
                ]
            else:
                lvl = "Low Successor Potential"; col = "#ff5757"
                dev = [
                    "Strengthen foundational competencies.",
                    "Improve peer collaboration & influence.",
                    "Enroll in capability-building programs."
                ]

        big_html += f"""
        <h2 style="margin-top:26px; font-size:28px; font-weight:800;">Succession Potential Indicator</h2>
        <div style="font-size:24px; font-weight:800; color:{col}; margin-top:8px;">{lvl}</div>
        <ul style="margin-top:8px; color:white; font-size:17px;">
        """
        for d in dev:
            big_html += f"<li style='margin-top:6px'>{d}</li>"
        big_html += "</ul>"

        big_html += "</div></div>"

        # render full HTML (scrolling)
        components.html(big_html, height=1450, scrolling=True)

    except Exception:
        st.error("Prediction failed.")
        st.code(traceback.format_exc())