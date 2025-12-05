import streamlit as st
import utils

def show_sidebar():
    with st.sidebar:
        st.title("Final Project")
        st.markdown("**MLC - CS365 - F1**")
        st.markdown("Acebes - Ewican - Milleza")
        
        st.markdown("### Navigation")

        # injected css cuz pasikat
        st.markdown(
            """
            <style>
            a.nav-btn {
                display: block;
                padding: 10px 15px;
                margin-bottom: 15px;
                margin-right: 5px;
                text-decoration: none !important;
                color: var(--text-color) !important;
                background-color: var(--background-color);
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                border: none;
                transition: all 0.2s ease;
                text-align: left;
                font-weight: 600;
            }
            a.nav-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 10px rgba(0,0,0,0.2);
                color: var(--primary-color) !important;
                text-decoration: none !important;
            }
            a.nav-btn:visited, a.nav-btn:active, a.nav-btn:focus {
                color: var(--text-color) !important;
                text-decoration: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <a class="nav-btn" href="#who-life-expectancy-project-overview"> Overview</a>
            <a class="nav-btn" href="#1-dataset-introduction"> 1. Dataset Introduction</a>
            <a class="nav-btn" href="#2-research-question"> 2. Research Question</a>
            <a class="nav-btn" href="#3-selected-analysis-techniques"> 3. Analysis Techniques</a>
            <a class="nav-btn" href="#4-data-exploration-preparation"> 4. Data Exploration</a>
            <a class="nav-btn" href="#5-analysis-and-insights"> 5. Analysis & Insights</a>
            <a class="nav-btn" href="#6-conclusions-and-recommendations"> 6. Conclusions</a>
            """,
            unsafe_allow_html=True
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
        st.metric("Timeframe", "2000-2015", "Annual data from countries")

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
    st.header("2. Research Questions")
    st.markdown(
        """
        Using the health and economic data in the dataset, we aimed to answer the following questions:
        """
    )
    st.subheader(
        """
        What are the most significant :green[economic] and public :red[health factors] that statistically determine a country's :blue[Life Expectancy?]
        """
    )
    st.subheader(
        """
        How can countries be :blue[segmented] into distinct public :orange[health groups] based on these metrics?
        """
    )
    st.info(
        """
        You can experiment with various data analysis techniques and data features to use in the next sections of the app.
        """
    )
    st.divider()
