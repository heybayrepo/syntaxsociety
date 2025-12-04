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

# -------------------------------------------------------------------------
# Promotion Eligibility Prediction — FINAL REPLACEMENT BLOCK (paste here)
# -------------------------------------------------------------------------
import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd

st.markdown("## 🧠 Predict Promotion Eligibility")

# --- load model safely ---
try:
    lr_model = joblib.load("logistic_pipeline.pkl")
except Exception:
    st.error("Model logistic_pipeline.pkl tidak ditemukan.")
    st.stop()

# --- input guard ---
if not emp or not isinstance(emp, (dict, pd.Series)):
    st.info("Isi data talent terlebih dahulu.")
    st.stop()

# --- helper ---
def safe_get(d, key, default=0.0):
    try:
        v = d.get(key, default)
        return float(v) if v is not None else default
    except:
        return default

# --- prepare data ---
input_dict = {
    "Age": safe_get(emp, "Age"),
    "Performance_Index": safe_get(emp, "Performance_Index"),
    "Leadership_Index": safe_get(emp, "Leadership_Index"),
    "Potential_Index": safe_get(emp, "Potential_Index"),
    "Training_Hours": safe_get(emp, "Training_Hours"),
    "Peer_Review_Score": safe_get(emp, "Peer_Review_Score"),
    "Projects_Handled": safe_get(emp, "Projects_Handled"),
    "Performance_Consistency": safe_get(emp, "Performance_Consistency"),
    "Growth_Momentum": safe_get(emp, "Growth_Momentum"),
}
input_df = pd.DataFrame([input_dict])

promotion_score = (
    0.30 * safe_get(emp, "Performance_Index") +
    0.25 * safe_get(emp, "Potential_Index") +
    0.20 * safe_get(emp, "Leadership_Index") +
    0.15 * safe_get(emp, "Performance_Consistency") +
    0.10 * safe_get(emp, "Growth_Momentum")
)

# threshold
try:
    df_promo = (
        0.30 * df["Performance_Index"] +
        0.25 * df["Potential_Index"] +
        0.20 * df["Leadership_Index"] +
        0.15 * df["Performance_Consistency"] +
        0.10 * df["Growth_Momentum"]
    )
    promo_threshold = df_promo.quantile(0.85)
except:
    promo_threshold = float("nan")

run_prediction = st.button("🔮 Predict Promotion Eligibility", key="predict_now_big")

if run_prediction:
    try:
        pred = int(lr_model.predict(input_df)[0])
        is_eligible = (pred == 1)
        status_color = "#00bf63" if is_eligible else "#ff5757"

        promo_str = f"{promotion_score:.2f}"
        thr_str = f"{promo_threshold:.2f}"

        # ============================
        # 1) WHY THIS RESULT LOGIC
        # ============================
        features_raw = [
            "Performance_Score","Leadership_Score","Peer_Review_Score",
            "Training_Hours","Projects_Handled","Performance_Consistency","Growth_Momentum"
        ]

        vals = {f: safe_get(emp, f) for f in features_raw}
        means = {}
        q75s = {}

        for f in features_raw:
            try:
                means[f] = float(df[f].mean())
                q75s[f] = float(df[f].quantile(0.75))
            except:
                means[f] = float("nan")
                q75s[f] = float("nan")

        strengths, weaknesses = [], []

        for f in features_raw:
            v, avg, q75 = vals[f], means[f], q75s[f]

            if np.isnan(avg):
                continue

            if is_eligible:
                if not np.isnan(q75) and v >= q75:
                    strengths.append((f, v, avg))
                elif v < avg:
                    weaknesses.append((f, v, avg))
            else:
                if v < avg:
                    weaknesses.append((f, v, avg))
                elif not np.isnan(q75) and v >= q75:
                    strengths.append((f, v, avg))

        # ============================
        # 2) TALENT RADAR CHART
        # ============================
        angles = np.linspace(0, 2*np.pi, len(features_raw), endpoint=False).tolist()
        angles += angles[:1]

        emp_plot = [vals[f] for f in features_raw] + [vals[features_raw[0]]]
        avg_plot = [
            means[f] if not np.isnan(means[f]) else 0 for f in features_raw
        ] + [means[features_raw[0]] if not np.isnan(means[features_raw[0]]) else 0]

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.plot(angles, avg_plot, color="#bbbbbb", linewidth=2, linestyle="dashed")
        ax.fill(angles, avg_plot, alpha=0.06, color="#bbbbbb")

        ax.plot(angles, emp_plot, color="#00bf63", linewidth=3)
        ax.fill(angles, emp_plot, alpha=0.20, color="#00bf63")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace("_", " ") for f in features_raw], color="white", fontsize=11)
        ax.grid(color="white", alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("white")

        buf = BytesIO()
        plt.savefig(buf, dpi=140, format="png", bbox_inches="tight", pad_inches=0.45)
        buf.seek(0)
        radar_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        # ============================
        # 3) SUCCESSION LOGIC
        # ============================
        thr = promo_threshold
        p = promotion_score

        if np.isnan(p) or np.isnan(thr):
            lvl, col = "Unknown", "#999"
            dev = ["Insufficient data."]
        else:
            if p >= thr:
                lvl, col = "High Successor Potential", "#00bf63"
                dev = [
                    "Provide advanced leadership exposure.",
                    "Start formal succession mentoring.",
                    "Assess readiness for expanded scope."
                ]
            elif p >= 0.9 * thr:
                lvl, col = "Emerging Successor", "#ffaa00"
                dev = [
                    "Start mid-level leadership coaching.",
                    "Increase cross-functional visibility.",
                    "Gradually expand responsibilities."
                ]
            else:
                lvl, col = "Low Successor Potential", "#ff5757"
                dev = [
                    "Strengthen foundational competencies.",
                    "Improve peer collaboration & influence.",
                    "Join capability-building programs."
                ]

        dev_items = "".join(
            [f"<li style='margin-top:6px;font-size:16px;color:white'>{d}</li>" for d in dev]
        )

        # ======================================================
        #  █████ ONE BIG CARD — ALL SECTIONS INSIDE ONE CARD ████
        # ======================================================
        big_html = f"""
<div style="background:#2e307d; border-radius:24px; padding:30px; width:100%;
            box-sizing:border-box; font-family:Inter, sans-serif; color:white;">

    <!-- PROMOTION PREDICTION -->
    <h2 style="font-size:32px; font-weight:800;">Promotion Prediction</h2>
    <div style="margin-top:10px; font-size:36px; font-weight:800; color:{status_color};">
        {'Eligible for Promotion' if is_eligible else 'Not Eligible'}
    </div>
    <div style="margin-top:10px; font-size:20px;">
        Promotion Score:
        <b style="color:{status_color}; font-size:24px;">{promo_str}</b>
        &nbsp;·&nbsp;
        Threshold (Q85):
        <b style="color:white; font-size:24px;">{thr_str}</b>
    </div>

    <!-- WHY THIS RESULT -->
    <h2 style="margin-top:28px; font-size:28px; font-weight:800;">Why This Result?</h2>
    <div style="margin-top:6px; font-size:17px;">
"""

        # weaknesses
        for f, v, avg in weaknesses:
            big_html += f"""
        <div style="margin-top:10px;">
            <b>{f.replace("_"," ")}:</b>
            <span style="color:#ff5757; font-weight:700">{v:.1f}</span>
            <span style="color:#bbb">(avg {avg:.1f})</span>
        </div>"""

        # strengths
        if strengths:
            big_html += "<div style='margin-top:18px; font-weight:700;'>Strengths:</div>"
            for f, v, avg in strengths:
                big_html += f"""
                <div style="margin-top:8px; font-size:17px;">
                    <b>{f.replace("_"," ")}:</b>
                    <span style="color:#00bf63; font-weight:700;">{v:.1f}</span>
                    <span style="color:#bbb;">(avg {avg:.1f})</span>
                </div>
                """

        # radar chart block
        big_html += f"""
    <!-- TALENT RADAR -->
    <h2 style="margin-top:32px; font-size:28px; font-weight:800;">Talent Radar Chart</h2>
    <div style="width:100%;border-radius:18px;overflow:hidden;margin-top:10px;">
        <img src="data:image/png;base64,{radar_b64}" style="width:100%;display:block;">
    </div>

    <!-- SUCCESSION -->
    <h2 style="margin-top:32px; font-size:28px; font-weight:800;">Succession Potential Indicator</h2>
    <div style="margin-top:10px; font-size:22px; font-weight:800; color:{col};">
        {lvl}
    </div>
    <div style="margin-top:10px; font-size:16px; color:#ddd;">Recommended Development Actions:</div>
    <ul style="margin-top:6px; padding-left:22px;">
        {dev_items}
    </ul>
</div>
"""

        components.html(f"<div class='full-width-container'>{big_html}</div>", height=1500, scrolling=False)

    except Exception as e:
        st.error("Error during prediction.")
        st.code(traceback.format_exc())