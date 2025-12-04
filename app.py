import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# =========================
# Load dataset
# =========================
df = pd.read_csv('data/Clean/dataset_clustered_dashboard.csv')

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
        total_talent = df["Employee_ID"].nunique()
        colA.markdown(overview_card("Total Talent", f"{total_talent}"), unsafe_allow_html=True)

    # ---- COLUMN B ----
    with colB:
        avg_age = df["Age"].mean()
        colB.markdown(overview_card("Average Talent Age", f"{avg_age:.1f} years"), unsafe_allow_html=True)

    # ============================
    # POSITION LEVEL CARD
    # ============================

    level_order = ["Junior", "Mid", "Senior", "Lead"]

    count_by_level = (
        df.groupby("Current_Position_Level")["Employee_ID"]
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
        sns.histplot(df["Age"], bins=12, color="#00bf63", ax=ax)
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
        sns.countplot(
            data=df,
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

    cluster_low = df[df["Cluster"] == 2].shape[0]
    cluster_high = df[df["Cluster"] == 4].shape[0]
    cluster_avg = df[df["Cluster"].isin([1, 3])].shape[0]

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


    import textwrap

    # ============================
    # POTENTIAL LOSS CARD
    # ============================

    salary_col = "Salary"   # sesuaikan jika nama kolom beda

    if salary_col not in df.columns:
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
        # total salary untuk cluster low
        s = pd.to_numeric(df.loc[df["Cluster"] == 2, salary_col], errors="coerce")
        total_salary = s.sum(skipna=True)

        # format Rp tanpa spasi terpisah
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
        position_options = ['All Levels'] + sorted(df['Current_Position_Level'].unique().tolist())
        selected_level = st.selectbox(
            'Filter by Position Level:',
            position_options,
            key="position_filter"
        )

    df_filtered = df if selected_level == 'All Levels' else df[df['Current_Position_Level'] == selected_level]

    if category == 'Best Performing':
        ranked = df_filtered.sort_values('Performance_Index', ascending=False).head(10)
        st.dataframe(
            ranked[
                ['Employee_ID', 'Current_Position_Level', 'Performance_Index',
                 'Performance_Consistency', 'Cluster']
            ],
            hide_index=True,
            use_container_width=True
        )

    elif category == 'Best Leadership':
        ranked = df_filtered.sort_values('Leadership_Index', ascending=False).head(10)
        st.dataframe(
            ranked[
                ['Employee_ID', 'Current_Position_Level', 'Leadership_Index',
                 'Leadership_Influence', 'Peer_Review_Score']
            ],
            hide_index=True,
            use_container_width=True
        )

    elif category == 'Best Potential':
        ranked = df_filtered.sort_values('Potential_Index', ascending=False).head(10)
        st.dataframe(
            ranked[
                ['Employee_ID', 'Current_Position_Level', 'Potential_Index',
                 'Growth_Momentum', 'Training_Hours']
            ],
            hide_index=True,
            use_container_width=True
        )

    # ⚖️ Average Indexes
    st.markdown("## ⚖️ Average Indexes")

    col1, col2, col3 = st.columns(3)

    # ===== change: metric_card returns HTML string (do not call st.markdown inside) =====
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

    # ===== display using .markdown (not .write) =====
    col1.markdown(metric_card('Performance Idx', f'{df["Performance_Index"].mean():.2f}'), unsafe_allow_html=True)
    col2.markdown(metric_card('Leadership Idx', f'{df["Leadership_Index"].mean():.2f}'), unsafe_allow_html=True)
    col3.markdown(metric_card('Potential Idx', f'{df["Potential_Index"].mean():.2f}'), unsafe_allow_html=True)


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
        position_options_risk = ['All Levels'] + sorted(df['Current_Position_Level'].unique().tolist())
        selected_level_risk = st.selectbox(
            'Filter by Position Level:',
            position_options_risk,
            key="risk_position_filter"
        )

    df_risk = df if selected_level_risk == 'All Levels' else df[df['Current_Position_Level'] == selected_level_risk]

    if risk_category == 'Low Performing':
        ranked = df_risk.sort_values('Performance_Index', ascending=True).head(10)
        st.dataframe(
            ranked[
                ['Employee_ID','Current_Position_Level','Performance_Index',
                 'Performance_Consistency','Cluster']
            ],
            hide_index=True, use_container_width=True
        )

    elif risk_category == 'Low Leadership':
        ranked = df_risk.sort_values('Leadership_Index', ascending=True).head(10)
        st.dataframe(
            ranked[
                ['Employee_ID','Current_Position_Level','Leadership_Index',
                 'Leadership_Influence','Peer_Review_Score']
            ],
            hide_index=True, use_container_width=True
        )

    else:
        ranked = df_risk.sort_values('Potential_Index', ascending=True).head(10)
        st.dataframe(
            ranked[
                ['Employee_ID','Current_Position_Level','Potential_Index',
                 'Growth_Momentum','Training_Hours']
            ],
            hide_index=True, use_container_width=True
        )

# ======================================================================
# TAB 3 — TALENT PREDICTOR (ASLI)
# ======================================================================
with tab3:

    # ensure a session-stored master dataframe so uploads persist during session
    if "master_df" not in st.session_state:
        st.session_state["master_df"] = globals().get("df", pd.DataFrame())

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

    if {"Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"}.issubset(df_ref.columns):
        cluster_map = (
            df_ref[["Cluster","Characteristics","Description","HR_Recommendations","HR_Programs"]]
            .drop_duplicates("Cluster")
            .set_index("Cluster")
            .to_dict("index")
        )
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
            # visually set dropdown to None to avoid confusion
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
            # safe extraction
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

        # Employee ID optional field
        emp_id_input = st.text_input("Employee ID (optional):", placeholder="e.g. NEW001", key="m_empid")

        # form fields (defaults are None — will show empty UI)
        age   = st.number_input("Age", min_value=18, max_value=70, value=None, key="m_age")
        perf  = st.selectbox("Performance Score (1–5)", ["Choose an option", 1,2,3,4,5], index=0, key="m_perf")
        lead  = st.number_input("Leadership Score", min_value=0.0, max_value=100.0, value=None, key="m_lead")
        train = st.number_input("Training Hours", min_value=0.0, max_value=500.0, value=None, key="m_train")
        proj  = st.number_input("Projects Handled", min_value=0.0, max_value=100.0, value=None, key="m_proj")
        peer  = st.number_input("Peer Review Score", min_value=0.0, max_value=100.0, value=None, key="m_peer")
        level = st.selectbox("Current Position Level", ["Choose an option","Junior","Mid","Senior","Lead"], index=0, key="m_level")

        # empty/placeholder view until all required fields (except optional emp id) are provided
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

        st.download_button("Download template CSV", template_df.to_csv(index=False), "employee_template.csv")
        uploaded = st.file_uploader("Upload CSV:", type=["csv"])

        if uploaded:
            try:
                new = pd.read_csv(uploaded)
                # basic validation: required columns presence
                required_cols = {"Employee_ID","Age","Performance_Score","Leadership_Score",
                                 "Training_Hours","Projects_Handled","Peer_Review_Score","Current_Position_Level"}
                if not required_cols.issubset(set(new.columns)):
                    st.error("Uploaded CSV missing required columns. Use the template.")
                else:
                    # append to session master df
                    st.session_state["master_df"] = pd.concat([st.session_state["master_df"], new], ignore_index=True)
                    df_ref = st.session_state["master_df"]
                    st.success(f"Uploaded {len(new)} rows. You can now search them in 'Select employee ID'.")
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")

        # in upload mode we stop here (no overview/cards)
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

    # ensure numeric keys exist on emp dict
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

    # lookup info safely
    if emp["Cluster"] is None:
        info = {"Characteristics":"—","Description":"—","HR_Recommendations":"—","HR_Programs":"—"}
    else:
        info = cluster_map.get(emp["Cluster"], {"Characteristics":"—","Description":"—","HR_Recommendations":"—","HR_Programs":"—"})

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
    # compute averages from df_ref if available otherwise 0
    avg_perf = df_ref["Performance_Index"].mean() if "Performance_Index" in df_ref.columns and not df_ref.empty else 0.0
    avg_lead = df_ref["Leadership_Index"].mean() if "Leadership_Index" in df_ref.columns and not df_ref.empty else 0.0
    avg_pot  = df_ref["Potential_Index"].mean() if "Potential_Index" in df_ref.columns and not df_ref.empty else 0.0

    c1,c2,c3 = st.columns(3)
    c1.markdown(
        small_card(
            "Performance Idx",
            f"{emp['Performance_Index']:.2f} | Avg {avg_perf:.2f}",
            "#00bf63" if emp["Performance_Index"] >= avg_perf else "#ff5757"
        ),
        unsafe_allow_html=True
    )

    c2.markdown(
        small_card(
            "Leadership Idx",
            f"{emp['Leadership_Index']:.2f} | Avg {avg_lead:.2f}",
            "#00bf63" if emp["Leadership_Index"] >= avg_lead else "#ff5757"
        ),
        unsafe_allow_html=True
    )

    c3.markdown(
        small_card(
            "Potential Idx",
            f"{emp['Potential_Index']:.2f} | Avg {avg_pot:.2f}",
            "#00bf63" if emp["Potential_Index"] >= avg_pot else "#ff5757"
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

    # ----------------------------------------------------------------------
    # Save back to globals for compatibility with rest of app (optional)
    # ----------------------------------------------------------------------
    globals()["df"] = st.session_state["master_df"]


# ----------------------------------------------------------------------
# Promotion Eligibility Predictor (Logistic Regression)
# ----------------------------------------------------------------------

import joblib
import numpy as np

st.markdown("## 🧠 Promotion Eligibility (Logistic Regression)")

# Load model
try:
    lr_model = joblib.load("logistic_pipeline.pkl")
except:
    st.error("Model logistic_pipeline.pkl tidak ditemukan. Train dulu sebelum menjalankan UI ini.")
    st.stop()

st.markdown("Isi data talent di bawah ini untuk memprediksi eligibility promosi.")

colA, colB, colC = st.columns(3)

with colA:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    perf_idx = st.number_input("Performance Index", min_value=0.0, max_value=100.0, value=70.0)
    lead_idx = st.number_input("Leadership Index", min_value=0.0, max_value=100.0, value=65.0)

with colB:
    pot_idx = st.number_input("Potential Index", min_value=0.0, max_value=100.0, value=68.0)
    training = st.number_input("Training Hours", min_value=0, max_value=500, value=40)
    peer = st.number_input("Peer Review Score", min_value=0.0, max_value=100.0, value=75.0)

with colC:
    projects = st.number_input("Projects Handled", min_value=0, max_value=100, value=5)
    consistency = st.number_input("Performance Consistency", min_value=0.0, max_value=100.0, value=80.0)
    growth = st.number_input("Growth Momentum", min_value=0.0, max_value=100.0, value=60.0)

# Prepare input array
input_features = np.array([[
    age,
    perf_idx,
    lead_idx,
    pot_idx,
    training,
    peer,
    projects,
    consistency,
    growth
]])

predict_btn = st.button("🔮 Predict Eligibility")

if predict_btn:
    pred = lr_model.predict(input_features)[0]
    proba = lr_model.predict_proba(input_features)[0][1]

    color = "#00bf63" if pred == 1 else "#ff5757"
    label = "Eligible for Promotion" if pred == 1 else "Not Eligible"

    st.markdown(
        f"""
        <div style="
            border:3px solid {color};
            border-radius:12px;
            padding:20px;
            margin-top:20px;
            background-color:#2e307d;
        ">
            <h3 style="color:white; margin:0; padding:0; font-size:26px;">
                Promotion Prediction
            </h3>

            <div style="margin-top:15px; font-size:32px; font-weight:700; color:{color};">
                {label}
            </div>

            <div style="margin-top:10px; font-size:18px; color:white;">
                Probability Score: <b>{proba:.3f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )