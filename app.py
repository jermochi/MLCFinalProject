import streamlit as st

st.set_page_config(page_title="MLC Final Project", page_icon="🌍", layout="wide")

st.title("WHO Life Expectancy Project Overview")
st.caption("The Influence of Life Expectancy Determinants: Understanding and Predicting Global Longevity Trends")
st.divider()

dataset_col, stats_col = st.columns([3, 2])
with dataset_col:
    st.header("1. 🌐 Dataset Introduction")
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
    st.metric("Rows", "≈ 3,000", "Country-Year pairs")
    st.metric("Columns", "22", "Health & economic features")

st.info(
    "The dataset gives us a away to analyze how socio-economic "
    "and public health factors work together to shape life expectancy."
)
st.divider()

st.header("2. ❓ Research Question")
st.header(
    """
    *What are the most significant economic and public health factors (e.g., GDP, immunization rates, mortality) that statistically determine a country's Life Expectancy, and how can countries be segmented into distinct public health groups based on these metrics?*
    """
)
st.divider()

st.header("3. 🛠️ Selected Analysis Techniques")
techniques = st.tabs(["Linear Regression", "K-means Clustering"])

with techniques[0]:
    st.subheader("Linear Regression")
    st.markdown(
        """
        - Models the relationship between predictors such as **GDP**, **Measles incidence**,
          and **Adult Mortality** against the target variable **Life Expectancy**.
        - Highlights statistically significant drivers, quantifying how each factor impacts longevity.
        """
    )

with techniques[1]:
    st.subheader("K-means Clustering")
    st.markdown(
        """
        - Groups countries into distinct clusters based on health and economic indicators.
        - Reveals natural clustering (e.g., high-GDP/low-disease vs. low-GDP/high-burden)
          to compare policy needs across clusters.
        """
    )

st.success("Together, these methods provide both predictive insight and clustering to show useable global information.")


#676767
