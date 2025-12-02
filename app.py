# ======================================================================
# 📌 Rakamin HR Intelligence Hub — Full App (3 Tabs + Stable Predictor)
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Rakamin HR Intelligence Hub", layout="wide")

# -----------------------------------------------------------
# LOAD INITIAL DATASET (adjust path if needed)
# -----------------------------------------------------------
try:
    df_init = pd.read_csv("data/Clean/dataset_clustered_dashboard.csv")
except Exception:
    df_init = pd.DataFrame()

# SESSION MASTER DF (for CSV uploads persistence)
if "master_df" not in st.session_state:
    st.session_state["master_df"] = df_init.copy()

df_ref = st.session_state["master_df"]

# -----------------------------------------------------------
# CSS / Styling
# -----------------------------------------------------------
css = """
<style>
[data-testid="stAppViewContainer"] { background-color: #449fe3; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -----------------------------------------------------------
# Reusable components
# -----------------------------------------------------------
def small_card(title, value, color="white", bg="#2e307d"):
    return f"""
    <div style="
        border:3px solid #2e307d;
        border-radius:12px;
        padding:16px;
        margin-bottom:12px;
        background-color:{bg};
        height:140px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        font-family:Inter, sans-serif;
    ">
        <h3 style="margin:0; padding:0; color:white; font-size:20px;">{title}</h3>
        <div style="margin-top:6px;">
            <span style="font-size:26px; font-weight:400; color:{color};">{value}</span>
        </div>
    </div>
    """

def build_description_card(text):
    return f"""
    <div style='width:100%; border:3px solid #2e307d; border-radius:12px;
                padding:20px; margin-top:10px; background-color:#2e307d;
                font-family:Inter, sans-serif;'>
        <div style='font-size:20px; font-weight:700; margin-bottom:8px; color:white;'>Description</div>
        <div style='font-size:16px; line-height:1.5; color:#dddddd;'>{text}</div>
    </div>
    """

def long_card(title, content):
    return f"""
    <div style='width:100%; border:3px solid #2e307d; border-radius:12px;
                padding:18px; margin-top:12px; background-color:#00bf63;
                font-family:Inter, sans-serif;'>
        <div style='font-size:20px; font-weight:700; margin-bottom:8px; color:white;'>{title}</div>
        <div style='font-size:15px; line-height:1.5; color:white;'>{content}</div>
    </div>
    """

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode()
    buf.close()
    return out

# -----------------------------------------------------------
# Title
# -----------------------------------------------------------
st.markdown("# **Rakamin HR Intelligence Hub**")
st.write("### *A centralized view of talent performance, potential, and predictive analytics.*")

# -----------------------------------------------------------
# Build cluster_map if available in df_ref
# -----------------------------------------------------------
if {"Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"}.issubset(df_ref.columns):
    cluster_map = (
        df_ref[["Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"]]
        .drop_duplicates("Cluster")
        .set_index("Cluster")
        .to_dict("index")
    )
else:
    cluster_map = {}

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Talent Overview", "⭐ Top & High Risk Talent", "🔮 Talent Predictor"])

# ======================================================================
# TAB 1 — Talent Overview (cards, counts, histograms, performance groups)
# ======================================================================
with tab1:
    st.header("📊 Talent Overview")

    # Top cards
    colA, colB = st.columns(2)
    with colA:
        total_talent = int(df_ref["Employee_ID"].nunique()) if not df_ref.empty and "Employee_ID" in df_ref.columns else 0
        st.markdown(small_card("Total Talent", total_talent), unsafe_allow_html=True)
    with colB:
        avg_age = df_ref["Age"].mean() if not df_ref.empty and "Age" in df_ref.columns else 0
        st.markdown(small_card("Average Talent Age", f"{avg_age:.1f} years"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Talent Count by Position Level
    st.markdown("### 📌 Talent Count by Position Level")
    level_order = ["Junior","Mid","Senior","Lead"]
    if "Current_Position_Level" in df_ref.columns and not df_ref.empty:
        counts = df_ref["Current_Position_Level"].value_counts().reindex(level_order, fill_value=0)
    else:
        counts = pd.Series([0,0,0,0], index=level_order)

    rows_html = "".join([
        f"""<div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter;">
                <div style="font-size:20px; font-weight:600; color:white;">{lvl}</div>
                <div style="font-size:20px; color:#dddddd;">{counts[lvl]} talent</div>
            </div>"""
        for lvl in level_order
    ])

    st.markdown(f"""
        <div style="border:3px solid #2e307d; border-radius:12px; padding:18px; background-color:#2e307d;">
            <div style="font-size:20px; font-weight:700; color:white; margin-bottom:10px;">Talent Count by Position Level</div>
            {rows_html}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Histograms: Age and Position Level
    st.markdown("### 📈 Talent Distribution")
    col1, col2 = st.columns(2)

    with col1:
        if "Age" in df_ref.columns and not df_ref.empty:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df_ref["Age"].dropna(), bins=12, ax=ax, color="#00bf63")
            ax.set_xlabel("Age", color="white"); ax.set_ylabel("Count", color="white")
            ax.tick_params(colors="white")
            fig.patch.set_facecolor("#2e307d"); ax.set_facecolor("#2e307d")
            st.image("data:image/png;base64," + fig_to_base64(fig))
            plt.close(fig)
        else:
            st.info("No Age data available.")

    with col2:
        if "Current_Position_Level" in df_ref.columns and not df_ref.empty:
            fig2, ax2 = plt.subplots(figsize=(6,3))
            sns.countplot(data=df_ref, x="Current_Position_Level", order=level_order, ax=ax2, color="#00bf63")
            ax2.set_xlabel("Position Level", color="white"); ax2.set_ylabel("Count", color="white")
            ax2.tick_params(colors="white")
            fig2.patch.set_facecolor("#2e307d"); ax2.set_facecolor("#2e307d")
            st.image("data:image/png;base64," + fig_to_base64(fig2))
            plt.close(fig2)
        else:
            st.info("No Position Level data available.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Performance Groups
    st.markdown("### 🧩 Performance Groups")
    if "Cluster" in df_ref.columns and not df_ref.empty:
        low = int((df_ref["Cluster"] == 2).sum())
        high = int((df_ref["Cluster"] == 4).sum())
        avg = int(df_ref["Cluster"].isin([1,3]).sum())
    else:
        low = high = avg = 0

    st.markdown(f"""
        <div style="border:3px solid #2e307d; border-radius:12px; padding:18px; background-color:#2e307d;">
            <div style="font-size:20px; font-weight:700; color:white; margin-bottom:8px;">Talent Count by Performance Group</div>
            <div style="display:flex; justify-content:space-between; padding:6px 0;"><div style="color:white;">Low Performing Talent</div><div style="color:#ff5757;">{low}</div></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0;"><div style="color:white;">High Performing Talent</div><div style="color:#00bf63;">{high}</div></div>
            <div style="display:flex; justify-content:space-between; padding:6px 0;"><div style="color:white;">Average Talent</div><div style="color:#dddddd;">{avg}</div></div>
        </div>
    """, unsafe_allow_html=True)

# ======================================================================
# TAB 2 — Top Talent & High Risk Talent
# ======================================================================
with tab2:
    st.header("⭐ Top Talent")

    colA, colB = st.columns(2)
    with colA:
        cat_top = st.selectbox("Select ranking category:", ["Best Performing", "Best Leadership", "Best Potential"], key="top_tab2_cat")
    with colB:
        pos_options_top = ["All Levels"] + (sorted(df_ref["Current_Position_Level"].dropna().unique().tolist()) if "Current_Position_Level" in df_ref.columns else [])
        level_top = st.selectbox("Filter by Position Level:", pos_options_top, key="top_tab2_level")

    df_top = df_ref.copy() if level_top == "All Levels" else df_ref[df_ref["Current_Position_Level"] == level_top]

    if cat_top == "Best Performing":
        ranked_top = df_top.sort_values("Performance_Index", ascending=False).head(10)
    elif cat_top == "Best Leadership":
        ranked_top = df_top.sort_values("Leadership_Index", ascending=False).head(10)
    else:
        ranked_top = df_top.sort_values("Potential_Index", ascending=False).head(10)

    st.dataframe(ranked_top, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.header("⚠️ High Risk Talent")

    colR1, colR2 = st.columns(2)
    with colR1:
        risk_cat = st.selectbox("Select category:", ["Low Performing", "Low Leadership", "Low Potential"], key="risk_tab2_cat")
    with colR2:
        pos_options_risk = ["All Levels"] + (sorted(df_ref["Current_Position_Level"].dropna().unique().tolist()) if "Current_Position_Level" in df_ref.columns else [])
        risk_level = st.selectbox("Filter by Position Level:", pos_options_risk, key="risk_tab2_level")

    df_risk = df_ref.copy() if risk_level == "All Levels" else df_ref[df_ref["Current_Position_Level"] == risk_level]

    if risk_cat == "Low Performing":
        ranked_risk = df_risk.sort_values("Performance_Index", ascending=True).head(10)
    elif risk_cat == "Low Leadership":
        ranked_risk = df_risk.sort_values("Leadership_Index", ascending=True).head(10)
    else:
        ranked_risk = df_risk.sort_values("Potential_Index", ascending=True).head(10)

    st.dataframe(ranked_risk, use_container_width=True, hide_index=True)

# ======================================================================
# TAB 3 — Talent Predictor (stable, session-safe CSV upload)
# ======================================================================
with tab3:
    st.header("🔮 Talent Predictor")
    st.markdown("### Select Talent Input Method")

    mode = st.radio(
        "",
        ["Select employee ID", "Predict employee cluster and characteristics", "Upload employee data in bulk using CSV"],
        horizontal=True,
        key="predictor_mode_tab3"
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
            ids = list(df_ref["Employee_ID"].dropna().astype(str).unique()) if not df_ref.empty and "Employee_ID" in df_ref.columns else []
            dropdown_vals = ["None"] + ids
            emp_dropdown = st.selectbox("Select Employee ID:", dropdown_vals, key="pred_tab3_select_dropdown")
        with colB:
            typed_id = st.text_input("Or type Employee ID:", placeholder="e.g. EMP0057", key="pred_tab3_typed_input")

        # typed id precedence
        if typed_id.strip() and typed_id.strip() in df_ref.get("Employee_ID", pd.Series(dtype=str)).astype(str).values:
            emp_id = typed_id.strip()
            try:
                st.session_state["pred_tab3_select_dropdown"] = "None"
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
    # 2) PREDICT — Manual Form
    # -----------------------------
    elif mode == "Predict employee cluster and characteristics":
        st.markdown("### Predict Employee Cluster and Characteristics")

        # add Employee ID field
        emp_id_input = st.text_input("Employee ID (optional):", placeholder="e.g. NEW001", key="pred_tab3_form_empid")

        age   = st.number_input("Age", min_value=18, max_value=70, value=None, key="pred_tab3_form_age")
        perf  = st.selectbox("Performance Score (1–5)", ["Choose an option",1,2,3,4,5], index=0, key="pred_tab3_form_perf")
        lead  = st.number_input("Leadership Score", min_value=0.0, max_value=100.0, value=None, key="pred_tab3_form_lead")
        train = st.number_input("Training Hours", min_value=0.0, max_value=500.0, value=None, key="pred_tab3_form_train")
        proj  = st.number_input("Projects Handled", min_value=0.0, max_value=100.0, value=None, key="pred_tab3_form_proj")
        peer  = st.number_input("Peer Review Score", min_value=0.0, max_value=100.0, value=None, key="pred_tab3_form_peer")
        level = st.selectbox("Current Position Level", ["Choose an option","Junior","Mid","Senior","Lead"], index=0, key="pred_tab3_form_level")

        required_filled = (
            (age is not None) and
            (perf != "Choose an option") and
            (lead is not None) and
            (train is not None) and
            (proj is not None) and
            (peer is not None) and
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
        st.download_button("Download template CSV", template_df.to_csv(index=False), "employee_template.csv", key="pred_tab3_dl_template")
        uploaded = st.file_uploader("Upload CSV:", type=["csv"], key="pred_tab3_file_uploader")
        if uploaded:
            try:
                new = pd.read_csv(uploaded)
                required_cols = {"Employee_ID","Age","Performance_Score","Leadership_Score","Training_Hours","Projects_Handled","Peer_Review_Score","Current_Position_Level"}
                if not required_cols.issubset(set(new.columns)):
                    st.error("Uploaded CSV missing required columns. Use the template.")
                else:
                    st.session_state["master_df"] = pd.concat([st.session_state["master_df"], new], ignore_index=True)
                    df_ref = st.session_state["master_df"]
                    st.success(f"Uploaded {len(new)} rows. Now search them in 'Select employee ID' mode.")
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

    # -----------------------------
    # EMPTY placeholder view
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
        cc1,cc2 = st.columns([0.28,0.72])
        cc1.markdown(small_card("Cluster", "—"), unsafe_allow_html=True)
        cc2.markdown(small_card("Characteristics", "—"), unsafe_allow_html=True)

        st.markdown(build_description_card("—"), unsafe_allow_html=True)
        st.markdown(long_card("HR Recommendations", "—"), unsafe_allow_html=True)
        st.markdown(long_card("Recommended Development Program", "—"), unsafe_allow_html=True)
        st.stop()

    # -----------------------------
    # FEATURE ENGINEERING (safe numeric)
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

    # -----------------------------
    # CLUSTERING (guarded)
    # -----------------------------
    if {"Performance_Index","Leadership_Index","Potential_Index","Cluster"}.issubset(df_ref.columns):
        centroids = df_ref.groupby("Cluster")[["Performance_Index","Leadership_Index","Potential_Index"]].mean()
        try:
            dist = ((centroids - [
                emp["Performance_Index"],
                emp["Leadership_Index"],
                emp["Potential_Index"]
            ])**2).sum(axis=1)
            emp["Cluster"] = int(dist.idxmin())
        except Exception:
            emp["Cluster"] = None
    else:
        emp["Cluster"] = None

    if emp["Cluster"] is None:
        info = {"Characteristics":"—","Description":"—","HR_Recommendations":"—","HR_Programs":"—"}
    else:
        info = cluster_map.get(emp["Cluster"], {"Characteristics":"—","Description":"—","HR_Recommendations":"—","HR_Programs":"—"})

    # -----------------------------
    # OVERVIEW CARDS
    # -----------------------------
    st.markdown("### Overview")
    oo1,oo2,oo3 = st.columns(3)
    oo1.markdown(small_card("Employee ID", emp.get("Employee_ID","—")), unsafe_allow_html=True)
    oo2.markdown(small_card("Age", emp.get("Age","—")), unsafe_allow_html=True)
    oo3.markdown(small_card("Position Level", emp.get("Current_Position_Level","—")), unsafe_allow_html=True)

    # -----------------------------
    # KEY TALENT INDEXES
    # -----------------------------
    st.markdown("### Key Talent Indexes")
    avg_perf = df_ref["Performance_Index"].mean() if "Performance_Index" in df_ref.columns and not df_ref.empty else 0.0
    avg_lead = df_ref["Leadership_Index"].mean() if "Leadership_Index" in df_ref.columns and not df_ref.empty else 0.0
    avg_pot  = df_ref["Potential_Index"].mean() if "Potential_Index" in df_ref.columns and not df_ref.empty else 0.0

    c1,c2,c3 = st.columns(3)
    c1.markdown(small_card("Performance Idx", f"{emp['Performance_Index']:.2f} | Avg {avg_perf:.2f}", "#00bf63" if emp["Performance_Index"] >= avg_perf else "#ff5757"), unsafe_allow_html=True)
    c2.markdown(small_card("Leadership Idx", f"{emp['Leadership_Index']:.2f} | Avg {avg_lead:.2f}", "#00bf63" if emp["Leadership_Index"] >= avg_lead else "#ff5757"), unsafe_allow_html=True)
    c3.markdown(small_card("Potential Idx", f"{emp['Potential_Index']:.2f} | Avg {avg_pot:.2f}", "#00bf63" if emp["Potential_Index"] >= avg_pot else "#ff5757"), unsafe_allow_html=True)

    # -----------------------------
    # CHARACTER & HR INSIGHTS
    # -----------------------------
    st.markdown("### Character")
    cc1,cc2 = st.columns([0.28,0.72])
    cc1.markdown(small_card("Cluster", emp.get("Cluster","—")), unsafe_allow_html=True)
    cc2.markdown(small_card("Characteristics", info.get("Characteristics","—")), unsafe_allow_html=True)

    st.markdown(build_description_card(info.get("Description","—")), unsafe_allow_html=True)
    st.markdown(long_card("HR Recommendations", info.get("HR_Recommendations","—")), unsafe_allow_html=True)
    st.markdown(long_card("Recommended Development Program", info.get("HR_Programs","—")), unsafe_allow_html=True)

    # Save back to session/global if necessary
    globals()["df"] = st.session_state["master_df"]

# ======================================================================
# End of app
# ======================================================================