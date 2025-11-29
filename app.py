import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


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

# === APPLY CSS (after definition) ===
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
# Section
# ========================= 

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# =========================
# Load dataset
# ========================= 

df = pd.read_csv('data/Clean/dataset_clustered_dashboard.csv')

# =========================
# Title and Description
# ========================= 

with header:
    st.markdown('# **Rakamin HR Intelligence Hub**')
    st.write('### *A centralized view of Rakamin workforce performance and potential.*')
    st.write('*Created by Syntax Society*')    

# ============================
# TALENT OVERVIEW — 2 COLUMN CARDS
# ============================

    st.markdown("## 📸 Talent Overview")

    colA, colB = st.columns(2)

    def overview_card(title, value):
        st.markdown(
        f"""
            <div style="
                border: 3px solid #2e307d;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 15px;
                background-color: #2e307d;
            ">
                <h3 style="margin:0; padding:0; color:white; font-size:22px;">{title}</h3>
                <div style="margin-top:10px;">
                    <span style="font-size:28px; font-weight:300; color:white;">
                        {value}
                    </span>
                </div>
            </div>
        """,
        unsafe_allow_html=True
        )


# ---- COLUMN A ----
with colA:
    total_talent = df["Employee_ID"].nunique()
    overview_card("Total Talent", f"{total_talent}")

# ---- COLUMN B ----
with colB:
    avg_age = df["Age"].mean()
    overview_card("Average Talent Age", f"{avg_age:.1f} years")


# ============================
# ROW 2 — SINGLE WIDE CARD
# ============================
# ------------------------------
# Talent Count by Position Level (fixed)
# ------------------------------

# desired order
level_order = ["Junior", "Mid", "Senior", "Lead"]

count_by_level = (
    df.groupby("Current_Position_Level")["Employee_ID"]
      .nunique()
      .reset_index()
      .rename(columns={"Employee_ID": "Total_Talent"})
)

# ensure all levels exist in the order (optional)
count_by_level = count_by_level[count_by_level["Current_Position_Level"].isin(level_order)]

# set categorical order and sort
count_by_level["Current_Position_Level"] = pd.Categorical(
    count_by_level["Current_Position_Level"],
    categories=level_order,
    ordered=True
)
count_by_level = count_by_level.sort_values("Current_Position_Level")

# build rows WITHOUT leading spaces/newlines (NO INDENTATION)
rows = []
for _, r in count_by_level.iterrows():
    level = r["Current_Position_Level"]
    total = int(r["Total_Talent"])
    rows.append(
        '<div style="display:flex;justify-content:space-between;padding:4px 0;'
        'border-bottom:1px solid rgba(255,255,255,0.05);font-family:Inter, sans-serif;">'
        f'<div style="font-size:22px;font-weight:600;color:white;">{level}</div>'
        f'<div style="font-size:22px;color:#dddddd;">{total} talent</div>'
        '</div>'
    )

rows_html = "".join(rows)

card_html = (
    '<div style="width:100%;border:3px solid #2e307d;border-radius:12px;'
    'padding:20px;margin-top:10px;background-color:#2e307d;'
    'font-family:Inter, sans-serif;">'
    '<div style="font-size:22px;font-weight:700;margin-bottom:15px;color:white;">'
    'Talent Count by Position Level'
    '</div>'
    + rows_html +
    '</div>'
)

st.markdown(card_html, unsafe_allow_html=True)

# ============================================
# QUARTILE PERFORMANCE CARD
# ============================================

# hitung quartiles
q1 = df['Performance_Index'].quantile(0.25)
q2 = df['Performance_Index'].quantile(0.50)
q3 = df['Performance_Index'].quantile(0.75)

count_q1 = df[df['Performance_Index'] <= q1].shape[0]
count_q4 = df[df['Performance_Index'] > q3].shape[0]
count_mid = df[(df['Performance_Index'] > q1) & (df['Performance_Index'] <= q3)].shape[0]

quartile_rows = f"""
<div style="display:flex; justify-content:space-between; padding:6px 0;
            border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter, sans-serif;">
    <div style="font-size:22px; font-weight:600; color:white;">Low Performers (Q1)</div>
    <div style="font-size:22px; color:#ff5757; font-weight:700;">{count_q1}</div>
</div>

<div style="display:flex; justify-content:space-between; padding:6px 0;
            border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter, sans-serif;">
    <div style="font-size:22px; font-weight:600; color:white;">High Performers (Q4)</div>
    <div style="font-size:22px; color:#00bf63; font-weight:700;">{count_q4}</div>
</div>

<div style="display:flex; justify-content:space-between; padding:6px 0;
            border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter, sans-serif;">
    <div style="font-size:22px; font-weight:600; color:white;">Normal Performers (Q2 & Q3)</div>
    <div style="font-size:22px; color:#dddddd; font-weight:500;">{count_mid}</div>
</div>
"""

quartile_card = f"""
<div style="
    width:100%;
    border:3px solid #2e307d;
    border-radius:12px;
    padding:20px;
    margin-top:15px;
    background-color:#2e307d;
    font-family:Inter, sans-serif;
">
    <div style="font-size:22px; font-weight:700; margin-bottom:12px; color:white;">
        Total Talent per Performance Quartiles
    </div>
    {quartile_rows}
</div>
"""

st.markdown(quartile_card, unsafe_allow_html=True)

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded

# ============================
# AGE & POSITION LEVEL DISTRIBUTION (SIDE BY SIDE)
# ============================

colH1, colH2 = st.columns(2)

# ============================
# AGE HISTOGRAM
# ============================
with colH1:
    fig, ax = plt.subplots(figsize=(5,3))
    sns.histplot(df['Age'], kde=False, bins=15, color="#00bf63", ax=ax)

    ax.set_title("Age Distribution", color="white")
    ax.set_xlabel("Age", color="white")
    ax.set_ylabel("Count", color="white")

    # background style
    ax.set_facecolor("#2e307d")
    fig.patch.set_facecolor("#2e307d")
    ax.tick_params(colors="white")

    age_img = plot_to_base64(fig)

    age_card = f"""
    <div style="
        width:100%;
        border:3px solid #2e307d;
        border-radius:12px;
        padding:20px;
        margin-top:15px;
        background-color:#2e307d;
        font-family:Inter, sans-serif;
    ">
        <div style="font-size:22px; font-weight:700; margin-bottom:12px; color:white;">
            Age Distribution
        </div>
        <img src="data:image/png;base64,{age_img}" style="width:100%; border-radius:10px;">
    </div>
    """

    st.markdown(age_card, unsafe_allow_html=True)


# ============================
# POSITION LEVEL DISTRIBUTION
# ============================
with colH2:
    fig, ax = plt.subplots(figsize=(5,3))

    order = ["Junior", "Mid", "Senior", "Lead"]
    sns.countplot(x=df["Current_Position_Level"], order=order, ax=ax, color="#00bf63")

    ax.set_title("Position Level Distribution", color="white")
    ax.set_xlabel("Position Level", color="white")
    ax.set_ylabel("Count", color="white")

    # styling
    ax.set_facecolor("#2e307d")
    fig.patch.set_facecolor("#2e307d")
    ax.tick_params(colors="white")
    for label in ax.get_xticklabels():
        label.set_color("white")

    pos_img = plot_to_base64(fig)

    pos_card = f"""
    <div style="
        width:100%;
        border:3px solid #2e307d;
        border-radius:12px;
        padding:20px;
        margin-top:15px;
        background-color:#2e307d;
        font-family:Inter, sans-serif;
    ">
        <div style="font-size:22px; font-weight:700; margin-bottom:12px; color:white;">
            Position Level Distribution
        </div>
        <img src="data:image/png;base64,{pos_img}" style="width:100%; border-radius:10px;">
    </div>
    """

    st.markdown(pos_card, unsafe_allow_html=True)




# =========================
# Top Talent
# =========================

st.markdown('## ⭐ Top Talent')

# ===========================================================
# TWO DROPDOWNS IN ONE ROW
# ===========================================================
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

# ===========================================================
# FILTER DATA BY POSITION LEVEL
# ===========================================================
if selected_level != 'All Levels':
    df_filtered = df[df['Current_Position_Level'] == selected_level]
else:
    df_filtered = df


# ===========================================================
# CATEGORY LOGIC (APPLY ON FILTERED DATA)
# ===========================================================

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

# =========================
# Average Indexes
# ========================= 

st.markdown('## ⚖️ Average Indexes')

col1, col2, col3 = st.columns(3)

# ================================
# Helper card wrapper
# ================================
def metric_card(title, value):
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True
    )

# ================================
# Column 1
# ================================
with col1:
    avg_performance = df['Performance_Index'].mean()
    metric_card('Performance', f'{avg_performance:.2f}')

# ================================
# Column 2
# ================================
with col2:
    avg_leadership = df['Leadership_Index'].mean()
    metric_card('Leadership', f'{avg_leadership:.2f}')

# ================================
# Column 3
# ================================
with col3:
    avg_potential = df['Potential_Index'].mean()
    metric_card('Potential', f'{avg_potential:.2f}')

# =========================
# High Risk Talent
# ========================= 

st.markdown('## ⚠️ High Risk Talent')

# ===========================================================
# TWO DROPDOWNS IN ONE ROW
# ===========================================================
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

# ===========================================================
# FILTER BY POSITION LEVEL
# ===========================================================
if selected_level_risk != 'All Levels':
    df_risk = df[df['Current_Position_Level'] == selected_level_risk]
else:
    df_risk = df

# ===========================================================
# APPLY CATEGORY SORTING ON FILTERED DATA
# ===========================================================
if risk_category == 'Low Performing':
    ranked = df_risk.sort_values('Performance_Index', ascending=True).head(10)
    st.dataframe(
        ranked[
            ['Employee_ID', 'Current_Position_Level', 'Performance_Index',
             'Performance_Consistency', 'Cluster']
        ],
        hide_index=True,
        use_container_width=True
    )

elif risk_category == 'Low Leadership':
    ranked = df_risk.sort_values('Leadership_Index', ascending=True).head(10)
    st.dataframe(
        ranked[
            ['Employee_ID', 'Current_Position_Level', 'Leadership_Index',
             'Leadership_Influence', 'Peer_Review_Score']
        ],
        hide_index=True,
        use_container_width=True
    )

elif risk_category == 'Low Potential':
    ranked = df_risk.sort_values('Potential_Index', ascending=True).head(10)
    st.dataframe(
        ranked[
            ['Employee_ID', 'Current_Position_Level', 'Potential_Index',
             'Growth_Momentum', 'Training_Hours']
        ],
        hide_index=True,
        use_container_width=True
    )

# =============================================
# TALENT HIGHLIGHT SECTION
# =============================================

st.markdown("## 🔎 Talent Selector")

# ======================================================
# 1. DEFINISIKAN small_card DULU (HARUS DI ATAS)
# ======================================================
def small_card(title, value):
    st.markdown(
        f"""
        <div style="
            border: 3px solid #2e307d;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 15px;
            background-color: #2e307d;
        ">
            <h3 style="margin:0; padding:0; color:white; font-size:20px;">{title}</h3>
            <div style="margin-top:10px;">
                <span style="font-size:24px; font-weight:400; color:white;">
                    {value}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# 2. DROPDOWN & INPUT MANUAL
# ======================================================
col1, col2 = st.columns([1, 1])

with col1:
    emp_dropdown = st.selectbox(
        "Select Employee ID:",
        df["Employee_ID"].unique(),
        key="dropdown_selector"
    )

with col2:
    emp_manual = st.text_input(
        "Or type Employee ID manually:",
        placeholder="e.g., EMP0057",
        key="manual_selector"
    )

# ======================================================
# 3. PRIORITIZE: manual input if valid
# ======================================================
if emp_manual.strip() in df["Employee_ID"].values:
    emp_id = emp_manual.strip()
else:
    emp_id = emp_dropdown

# ======================================================
# 4. LOAD SELECTED EMPLOYEE
# ======================================================
emp = df[df["Employee_ID"] == emp_id].iloc[0]
# =============================================
# 2. CARD ROW 1 (Age, Position Level)
# =============================================

st.markdown('### Overview')

col1, col2 = st.columns(2)

with col1:
    small_card("Age", f"{int(emp['Age'])}")

with col2:
    small_card("Position Level", f"{emp['Current_Position_Level']}")

# =============================================
# 3. MEAN METRICS — COMPANY AVERAGES
# =============================================

avg_performance = df["Performance_Index"].mean()
avg_leadership = df["Leadership_Index"].mean()
avg_potential = df["Potential_Index"].mean()
avg_projects = df["Projects_Handled"].mean()
avg_peer = df["Peer_Review_Score"].mean()

# =============================================
# 4. CARD ROW 2 — 5 METRIC CARDS
# =============================================

# =========================================
# INDIVIDUAL VS COMPANY AVERAGE — CARD VIEW
# =========================================

# compute all metrics needed
metrics = [
    ("Performance Index", emp["Performance_Index"], df["Performance_Index"].mean()),
    ("Leadership Index", emp["Leadership_Index"], df["Leadership_Index"].mean()),
    ("Potential Index", emp["Potential_Index"], df["Potential_Index"].mean()),
    ("Projects Handled", emp["Projects_Handled"], df["Projects_Handled"].mean()),
]

# build rows HTML (compact padding)
rows = []
for title, val, avg in metrics:
    # warna dinamis: merah kalau karyawan < rata-rata
    color = "#ff5757" if val < avg else "#dddddd"

    rows.append(
        f'<div style="display:flex; justify-content:space-between; padding:6px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.06); font-family:Inter, sans-serif;">'
        f'<div style="font-size:22px; font-weight:600; color:white;">{title}</div>'
        f'<div style="font-size:22px; color:{color};">{val:.1f}  |  Avg {avg:.1f}</div>'
        '</div>'
    )

rows_html = "".join(rows)

# card wrapper — identical style to Talent Count card
card_html = (
    '<div style="width:100%; border:3px solid #2e307d; border-radius:12px; '
    'padding:20px; margin-top:10px; background-color:#2e307d; '
    'font-family:Inter, sans-serif;">'
    '<div style="font-size:22px; font-weight:700; margin-bottom:15px; color:white;">'
    'Individual vs Company Average'
    '</div>'
    + rows_html +
    '</div>'
)

st.markdown(card_html, unsafe_allow_html=True)
st.markdown('*Note: Values in red indicate below company average.*')

# ======================================================
# CARD TEMPLATE: small (two-column)
# ======================================================

def info_card(title, value):
    st.markdown(
        f"""
        <div style="
            border:3px solid #2e307d;
            border-radius:12px;
            padding:16px;
            margin-bottom:15px;
            background-color:#2e307d;
        ">
            <h3 style="margin:0; padding:0; color:white; font-size:20px;">
                {title}
            </h3>
            <div style="margin-top:10px;">
                <span style="font-size:22px; font-weight:400; color:white;">
                    {value}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# ROW: CLUSTER + CHARACTERISTICS
# ======================================================

st.markdown('### Character')

colC1, colC2 = st.columns(2)

with colC1:
    info_card("Cluster", emp["Cluster"])

with colC2:
    info_card("Characteristics", emp["Characteristics"])

# ======================================================
# FULL-WIDTH DESCRIPTION CARD
# ======================================================

desc_html = f"""
<div style="
    width:100%;
    border:3px solid #2e307d;
    border-radius:12px;
    padding:20px;
    margin-top:10px;
    background-color:#2e307d;
    font-family:Inter, sans-serif;
">
    <div style="font-size:22px; font-weight:700; margin-bottom:15px; color:white;">
        Description
    </div>
    <div style="font-size:20px; color:#dddddd; line-height:1.5;">
        {emp['Description']}
    </div>
</div>
"""

st.markdown(desc_html, unsafe_allow_html=True)

st.markdown('## 💡 HR Insight')

# =========================================
# CLEAN TEXT (simple)
# =========================================
def clean_text_simple(raw):
    if pd.isna(raw):
        return ""
    return str(raw).replace("\n", "<br>").strip()

hr_text = clean_text_simple(emp["HR_Recommendations"])
program_text = clean_text_simple(emp["HR_Programs"])

# =========================================
# CARD TEMPLATE FUNCTION
# =========================================
def long_card(title, content):
    return (
        '<div style="width:100%; border:3px solid #2e307d; border-radius:12px;'
        ' padding:20px; margin-top:20px; background-color:#00bf63;'
        ' font-family:Inter, sans-serif;">'
            f'<div style="font-size:22px; font-weight:700; margin-bottom:12px; color:white;">{title}</div>'
            f'<div style="font-size:20px; line-height:1.6; color:white;">{content}</div>'
        '</div>'
    )

# =========================================
# RENDER CARDS
# =========================================
st.markdown(long_card("HR Recommendations", hr_text), unsafe_allow_html=True)
st.markdown(long_card("Recommended Development Program", program_text), unsafe_allow_html=True)