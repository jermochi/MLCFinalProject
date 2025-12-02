import streamlit as st
import utils

def show_sidebar():
    with st.sidebar:
        st.title("Navigation")
        st.markdown(
            """
            - [Overview](#who-life-expectancy-project-overview)
            - [1. Dataset Introduction](#1-dataset-introduction)
            - [2. Research Question](#2-research-question)
            - [3. Analysis Techniques](#3-selected-analysis-techniques)
            - [4. Data Exploration](#4-data-exploration-preparation)
            - [5. Analysis & Insights](#5-analysis-and-insights)
            - [6. Conclusions](#6-conclusions-and-recommendations)
            """
        )
        st.caption("Use links to jump to section")

def show_overview():
    st.title("WHO Life Expectancy Project Overview")
    st.caption("The Influence of Life Expectancy Determinants: Understanding and Predicting Global Longevity Trends")
    st.divider()

def show_dataset_introduction(df):
    dataset_col, stats_col = st.columns([3, 2])
    with dataset_col:
        st.header("1. Dataset Introduction")
        st.markdown(
            """
            We use the **WHO Life Expectancy Dataset**, a public health resource
            combining health and economic indicators for more than **190 countries** across
            multiple years. This dataset has been cleaned to get rid of noise and invalid inputs.
            
            - Each row = a **country-year** observation capturing metrics required to study
              the factors of human longevity.
            - **Source**: Life Expectancy (WHO) Fixed Dataset via Kaggle 
            """
        )
    with stats_col:
        st.metric("Rows", "approx. 3,000", "Country-Year pairs")
        st.metric("Columns", "22", "Health & economic features")

    st.info(
        "The dataset gives us a away to analyze how socio-economic "
        "and public health factors work together to shape life expectancy."
    )
    
    @st.dialog("Explore Dataset Structure & Columns", width="large")
    def show_dataset_details():
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Dataset Structure")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.subheader("Column Details")
        
        col_info = utils.get_column_info(df)
        st.dataframe(col_info, hide_index=True, use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.write(df.describe())

    if st.button("View Dataset Details", use_container_width=True):
        show_dataset_details()
    st.divider()

def show_research_question():
    st.header("2. Research Question")
    st.header(
        """
        *What are the most significant economic and public health factors (e.g., GDP, immunization rates, mortality) that statistically determine a country's Life Expectancy, and how can countries be segmented into distinct public health groups based on these metrics?*
        """
    )
    st.divider()
