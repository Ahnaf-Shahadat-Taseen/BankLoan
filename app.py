import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

# 1. Page Config
st.set_page_config(page_title="Dynamic Loan Analytics", page_icon="🏦", layout="wide")

# 2. Load Model, Scaler, and Data
@st.cache_resource
def load_assets():
    if os.path.exists('loan_model.pkl') and os.path.exists('scaler.pkl'):
        return joblib.load('loan_model.pkl'), joblib.load('scaler.pkl')
    return None, None

@st.cache_data
def load_viz_data():
    if os.path.exists('loan_approval_dataset.csv'):
        df = pd.read_csv('loan_approval_dataset.csv')
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        return df
    return None

model, scaler = load_assets()
df_viz = load_viz_data()

st.title("🏦 Bank Loan Approval System & Dynamic Analytics")
st.divider()

# --- PREDICTOR SECTION ---
if model is not None:
    st.header("🔍 Loan Eligibility Checker")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            dependents = st.number_input("Dependents", 0, 10, 2)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income = st.number_input("Annual Income", value=500000)
            loan_amt = st.number_input("Loan Amount", value=1000000)
        with c2:
            term = st.number_input("Term (Years)", 1, 25, 10)
            cibil = st.number_input("CIBIL Score", 300, 900, 750)
            res = st.number_input("Residential Asset", value=1000000)
            com = st.number_input("Commercial Asset", value=500000)
            lux = st.number_input("Luxury Asset", value=200000)
            bank = st.number_input("Bank Asset", value=300000)
        
        if st.form_submit_button("Predict Status"):
            edu_v = 1 if education == "Graduate" else 0
            emp_v = 1 if self_employed == "Yes" else 0
            feat = np.array([[dependents, edu_v, emp_v, income, loan_amt, term, cibil, res, com, lux, bank]])
            pred = model.predict(scaler.transform(feat))
            if pred[0] == 1: st.success("✅ Loan Approved!")
            else: st.error("❌ Loan Rejected!")

st.divider()

# --- DYNAMIC ANALYTICS SECTION ---
if df_viz is not None:
    st.header("📊 Dynamic Data Exploration")
    st.markdown("Select columns from the dropdowns to update the charts automatically.")

    col_charts_1, col_charts_2 = st.columns(2)

    with col_charts_1:
        st.subheader("Distribution Analysis")
        # ড্রপডাউন ১: কলাম সিলেক্ট করা (যেমন: Income, CIBIL, Loan Amount)
        dist_col = st.selectbox("Select Column for Distribution:", 
                                ['cibil_score', 'income_annum', 'loan_amount', 'loan_term'])
        
        fig_hist = px.histogram(df_viz, x=dist_col, color="loan_status", 
                                 marginal="box", nbins=30,
                                 title=f"Distribution of {dist_col}",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_charts_2:
        st.subheader("Comparison Analysis")
        # ড্রপডাউন ২: ক্যাটাগরি সিলেক্ট করা (Education, Self Employed)
        cat_col = st.selectbox("Compare Approval Status by:", 
                                ['education', 'self_employed', 'no_of_dependents'])
        
        fig_bar = px.histogram(df_viz, x=cat_col, color="loan_status", 
                                barmode="group",
                                title=f"Loan Status by {cat_col}",
                                color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    
    col_charts_3, col_charts_4 = st.columns(2)

    with col_charts_3:
        st.subheader("Relationship Analysis (Scatter)")
        # ড্রপডাউন ৩ ও ৪: X এবং Y অক্ষ সিলেক্ট করা
        x_axis = st.selectbox("X-axis:", ['income_annum', 'cibil_score', 'loan_amount'])
        y_axis = st.selectbox("Y-axis:", ['loan_amount', 'bank_asset_value', 'luxury_assets_value'])
        
        fig_scatter = px.scatter(df_viz, x=x_axis, y=y_axis, color="loan_status",
                                  title=f"{x_axis} vs {y_axis}",
                                  opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_charts_4:
        st.subheader("Asset Breakdown")
        asset_option = st.multiselect("Select Assets to Compare:", 
                                      ['residential_assets_value', 'commercial_assets_value', 
                                       'luxury_assets_value', 'bank_asset_value'],
                                      default=['residential_assets_value', 'commercial_assets_value'])
        
        df_melt = df_viz.groupby('loan_status')[asset_option].mean().reset_index().melt(id_vars='loan_status')
        fig_asset = px.bar(df_melt, x='variable', y='value', color='loan_status', barmode='group',
                            title="Average Asset Value by Loan Status")
        st.plotly_chart(fig_asset, use_container_width=True)

else:
    st.warning("Please ensure 'loan_approval_dataset.csv' is in the same folder.")

st.caption("Developed by Ahnaf Shahadat Taseen")