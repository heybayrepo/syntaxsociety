import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #449fe3;  /* ganti warna sesuai kebutuhan */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.markdown('# **HR Intelligence Hub**')
st.write('### *A centralized view of Rakamin workforce performance and potential.*')
st.write('Created by Syntax Society')    


# =========================
# Load dataset final
# =========================
df = pd.read_csv("dataset_clustered_final.csv")

# Pastikan kolom bersih dan konsisten
df.columns = df.columns.str.strip()

# Pilih kolom identitas
# Jika dataset punya Name, gunakan Name. Kalau tidak, pakai Employee_ID.
identifier_col = "Employee_ID"
if "Name" in df.columns:
    identifier_col = "Name"

# =========================
# Employee Selector
# =========================
st.subheader("Employee Selector")

selected_id = st.selectbox(
    "Select Employee ID",
    df[identifier_col].unique()
)

# Ambil baris employee yang dipilih
selected_employee = df[df[identifier_col] == selected_id].iloc[0]

# Konfirmasi pilihan
st.markdown(
    f"""
    <div style="padding:10px; background:#2e307d; border-radius:8px; margin-top:5px;">
        <b>Selected Employee:</b> {selected_id}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Cluster Profile Summary
# =========================

st.subheader("Profile Summary")

cluster_id = int(selected_employee["Cluster"])
cluster_desc = str(selected_employee["Characteristics"])

with st.container():
    st.markdown(
        f"""
        <div style="
            padding:10px 16px;
            border-radius:10px;
            background-color:#2e307d;  
            border:1px solid rgba(255,255,255,0.2);
            margin-top:10px;
            display:block;
            width:fit-content;
        ">
            <h4 style="
                margin:0;
                padding:0;
                font-size:18px;
                font-weight:600;
                color:#ffffff;
            ">
                Cluster {cluster_id}
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Deskripsi ditampilkan terpisah agar aman
    st.markdown("**Description:**")
    st.write(cluster_desc)