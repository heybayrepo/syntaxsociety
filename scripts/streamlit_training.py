import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #449fe3;  /* ganti warna sesuai kebutuhan */
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

df = pd.read_csv('dataset_clustered_dashboard.csv')

# =========================
# Title and Description
# ========================= 

with header:
    st.markdown('# **Welcome to HR Intelligence Hub**')
    st.write('### *A centralized view of Rakamin workforce performance and potential.*')
    st.write('*Created by Syntax Society*')    

with dataset:
    st.markdown('### **Rakamin Talent Promotion**')
    st.dataframe(df)