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

import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# Function: convert matplotlib figure ➜ base64 (so it fits inside HTML card)
def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    buffer.seek(0)
    img_bytes = buffer.read()
    buffer.close()
    return base64.b64encode(img_bytes).decode()

# ============================
# HISTOGRAM CARDS (NO TITLE, WITH SPACING)
# ============================

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# ======== AGE DISTRIBUTION ========
with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df["Age"], bins=12, color="#00bf63", ax=ax)

    ax.set_title("")
    ax.set_xlabel("Age", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#2e307d")
    ax.set_facecolor("#2e307d")

    img_age = fig_to_base64(fig)

    st.markdown(
        f"""
        <div style="
            border:3px solid #2e307d;
            border-radius:12px;
            padding:20px;
            margin-top:20px;
            background-color:#2e307d;">
            <img src="data:image/png;base64,{img_age}" style="width:100%; border-radius:10px;" />
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close(fig)

# ======== POSITION LEVEL DISTRIBUTION ========
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.countplot(
        data=df,
        x="Current_Position_Level",
        order=["Junior", "Mid", "Senior", "Lead"],
        color="#00bf63",
        ax=ax2,
    )

    ax2.set_title("")
    ax2.set_xlabel("Position Level", color="white")
    ax2.set_ylabel("Count", color="white")
    ax2.tick_params(colors="white")
    fig2.patch.set_facecolor("#2e307d")
    ax2.set_facecolor("#2e307d")

    img_pos = fig_to_base64(fig2)

    st.markdown(
        f"""
        <div style="
            border:3px solid #2e307d;
            border-radius:12px;
            padding:20px;
            margin-top:20px;
            background-color:#2e307d;">
            <img src="data:image/png;base64,{img_pos}" style="width:100%; border-radius:10px;" />
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

cluster_rows = """
<div style='display:flex; justify-content:space-between; padding:6px 0;
            border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter, sans-serif;'>
    <div style='font-size:22px; font-weight:600; color:white;'>Low Performing Talent</div>
    <div style='font-size:22px; font-weight:700; color:#ff5757;'>""" + str(cluster_low) + """</div>
</div>

<div style='display:flex; justify-content:space-between; padding:6px 0;
            border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter, sans-serif;'>
    <div style='font-size:22px; font-weight:600; color:white;'>High Performing Talent</div>
    <div style='font-size:22px; font-weight:700; color:#00bf63;'>""" + str(cluster_high) + """</div>
</div>

<div style='display:flex; justify-content:space-between; padding:6px 0;
            border-bottom:1px solid rgba(255,255,255,0.08); font-family:Inter, sans-serif;'>
    <div style='font-size:22px; font-weight:600; color:white;'>Average Talent</div>
    <div style='font-size:22px; font-weight:500; color:#dddddd;'>""" + str(cluster_avg) + """</div>
</div>
"""

cluster_card = """
<div style='width:100%; border:3px solid #2e307d; border-radius:12px;
            padding:20px; margin-top:15px; background-color:#2e307d;
            font-family:Inter, sans-serif;'>
    <div style='font-size:22px; font-weight:700; margin-bottom:12px; color:white;'>
        Talent Count by Performance Group
    </div>
    """ + cluster_rows + """
</div>
"""

st.markdown(cluster_card, unsafe_allow_html=True)

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
# 🔮 TALENT PREDICTOR — REVISED VERSION
# =============================================

st.markdown("## 🔎 Talent Predictor")

# ======================================================
# 0. Load cluster → HR mapping (once)
# ======================================================
cluster_map = (
    df[["Cluster","Characteristics","Description",
        "HR_Recommendations","HR_Programs"]]
    .drop_duplicates("Cluster")
    .set_index("Cluster")
    .to_dict("index")
)

# Card style
def small_card(title, value, color="white"):
    return f"""
    <div style="
        border:3px solid #2e307d;
        border-radius:12px;
        padding:16px;
        margin-bottom:15px;
        background-color:#2e307d;
    ">
        <div style="font-size:20px; font-weight:600; color:white;">
            {title}
        </div>
        <div style="margin-top:10px; font-size:24px; color:{color}; font-weight:500;">
            {value}
        </div>
    </div>
    """

# ======================================================
# 1. INPUT MODE
# ======================================================
st.markdown("### Select Talent Input Method")

mode = st.radio(
    "",
    ["Select current employee", "Add new employee manually", "Upload employee data in bulk using CSV"],
    horizontal=True
)

# ======================================================
# 2. MANUAL INPUT
# ======================================================
def manual_form():
    st.markdown("### Manual Input")

    age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
    perf = perf = st.selectbox(
        "Performance Score (1–5)",
        [1, 2, 3, 4, 5]
    )
    lead = st.number_input("Leadership Score", 0, 100, value=50, step=1)
    train = st.number_input("Training Hours", 0, 500, value=20, step=1)
    proj = st.number_input("Projects Handled", 0, 100, value=5, step=1)
    peer = st.number_input("Peer Review Score", 0, 100, value=60, step=1)

    level = st.selectbox(
        "Current Position Level",
        ["Junior","Mid","Senior","Lead"]
    )

    return {
        "Age": age,
        "Performance_Score": perf,
        "Leadership_Score": lead,
        "Training_Hours": train,
        "Projects_Handled": proj,
        "Peer_Review_Score": peer,
        "Current_Position_Level": level
    }

# ======================================================
# 3. UPLOAD CSV
# ======================================================
template_df = pd.DataFrame({
    "Employee_ID":["EMP0001"],
    "Age":[30],
    "Performance_Score":[80],
    "Leadership_Score":[70],
    "Training_Hours":[40],
    "Projects_Handled":[5],
    "Peer_Review_Score":[85],
    "Current_Position_Level":["Senior"]
})
csv_template = template_df.to_csv(index=False)

uploaded_df = None

if mode == "Upload employee data in bulk using CSV":
    st.markdown("*Note: Please follow this exact format.*")
    st.download_button("Download template CSV", csv_template, "employee_template.csv")

    uploaded_file = st.file_uploader("Upload CSV:", type=["csv"])
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)

# ======================================================
# 4. DETERMINE EMPLOYEE INPUT
# ======================================================
emp = None

if mode == "Select current employee":
    emp_id = st.selectbox("Select Employee ID:", df["Employee_ID"].unique())
    emp = df[df["Employee_ID"] == emp_id].iloc[0].to_dict()

elif mode == "Add new employee manually":
    emp = manual_form()

elif mode == "Upload employee data in bulk using CSV":
    if uploaded_df is not None and len(uploaded_df) > 0:
        emp = uploaded_df.iloc[0].to_dict()
    else:
        st.stop()

if emp is None:
    st.warning("Please provide employee data.")
    st.stop()

# ======================================================
# 5. FEATURE ENGINEERING
# ======================================================
emp["Leadership_Index"] = 0.4*emp["Leadership_Score"] + 0.6*emp["Peer_Review_Score"]
emp["Performance_Index"] = (
    0.5*emp["Performance_Score"] +
    0.2*emp["Projects_Handled"] +
    0.3*emp["Peer_Review_Score"]
)
emp["Potential_Index"] = (
    0.4*emp["Training_Hours"] +
    0.4*emp["Peer_Review_Score"] +
    0.2*emp["Leadership_Score"]
)

# ======================================================
# 6. CLUSTER PREDICTION
# ======================================================
centroids = df.groupby("Cluster")[["Performance_Index","Leadership_Index","Potential_Index"]].mean()
dist = ((centroids - [
    emp["Performance_Index"],
    emp["Leadership_Index"],
    emp["Potential_Index"]
])**2).sum(axis=1)

emp["Cluster"] = dist.idxmin()

# ======================================================
# 7. HR INFO
# ======================================================
info = cluster_map.get(emp["Cluster"])

# ======================================================
# 8. OVERVIEW
# ======================================================
st.markdown("### Overview")
c1, c2 = st.columns(2)

c1.markdown(small_card("Age", int(emp["Age"])), unsafe_allow_html=True)
c2.markdown(small_card("Position Level", emp["Current_Position_Level"]), unsafe_allow_html=True)

# ======================================================
# 9. INDEX CARDS — ***3 INDEX ONLY, IN 1 ROW (3 COLUMNS)***
# ======================================================
st.markdown("### Key Talent Indexes")

avg_perf  = df["Performance_Index"].mean()
avg_lead  = df["Leadership_Index"].mean()
avg_pot   = df["Potential_Index"].mean()

i_col1, i_col2, i_col3 = st.columns(3)

# Performance
color_perf = "#00bf63" if emp["Performance_Index"] >= avg_perf else "#ff5757"
i_col1.markdown(
    small_card(
        "Performance Index",
        f"{emp['Performance_Index']:.2f} | Avg {avg_perf:.2f}",
        color=color_perf
    ),
    unsafe_allow_html=True
)

# Leadership
color_lead = "#00bf63" if emp["Leadership_Index"] >= avg_lead else "#ff5757"
i_col2.markdown(
    small_card(
        "Leadership Index",
        f"{emp['Leadership_Index']:.2f} | Avg {avg_lead:.2f}",
        color=color_lead
    ),
    unsafe_allow_html=True
)

# Potential
color_pot = "#00bf63" if emp["Potential_Index"] >= avg_pot else "#ff5757"
i_col3.markdown(
    small_card(
        "Potential Index",
        f"{emp['Potential_Index']:.2f} | Avg {avg_pot:.2f}",
        color=color_pot
    ),
    unsafe_allow_html=True
)

# ======================================================
# 10. CLUSTER + CHARACTERISTICS
# ======================================================
st.markdown("### Character")

cc1, cc2 = st.columns(2)
cc1.markdown(small_card("Cluster", emp["Cluster"]), unsafe_allow_html=True)
cc2.markdown(small_card("Characteristics", info["Characteristics"]), unsafe_allow_html=True)

# ======================================================
# 11. DESCRIPTION + HR RECOMMENDATION
# ======================================================
desc_card = f"""
<div style="
    width:100%; border:3px solid #2e307d; border-radius:12px;
    padding:20px; margin-top:20px; background-color:#2e307d;
">
    <div style="font-size:22px; font-weight:700; margin-bottom:12px; color:white;">
        Description
    </div>
    <div style="font-size:18px; line-height:1.5; color:#dddddd;">
        {info["Description"]}
    </div>
</div>
"""
st.markdown(desc_card, unsafe_allow_html=True)

def long_card(title, content):
    return f"""
    <div style='width:100%; border:3px solid #2e307d; border-radius:12px;
                padding:20px; margin-top:20px; background-color:#00bf63;'>
        <div style='font-size:22px; font-weight:700; margin-bottom:12px; color:white;'>{title}</div>
        <div style='font-size:18px; line-height:1.6; color:white;'>{content}</div>
    </div>
    """

st.markdown(long_card("HR Recommendations", info["HR_Recommendations"]), unsafe_allow_html=True)
st.markdown(long_card("Recommended Development Program", info["HR_Programs"]), unsafe_allow_html=True)