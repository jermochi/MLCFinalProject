import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
import utils
import plots
import io

st.set_page_config(page_title="MLC Final Project", page_icon="🌍", layout="wide")

# --- Caching Expensive Operations ---
@st.cache_data
def get_correlation_heatmap(df):
    return plots.plot_correlation_heatmap(df)

@st.cache_data
def get_distribution_plot(df):
    return plots.plot_life_expectancy_distribution(df)

@st.cache_data
def train_regression_model(df, selected_features):
    X = df[selected_features]
    y = df['Life_expectancy']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, X, y

@st.cache_data
def get_regression_plot(y, y_pred):
    return plots.plot_actual_vs_predicted(y, y_pred)

@st.cache_data
def get_kmeans_clusters(df, cluster_features, k):
    X_cluster = df[cluster_features]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    return clusters

@st.cache_data
def get_cluster_plot(df, x_axis, y_axis):
    return plots.plot_clusters(df, x_axis, y_axis)

def render_centered_plot(fig):
    """Renders a plot centered and smaller, clickable to expand."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(buf, use_container_width=True)

# Project Overview
st.title("WHO Life Expectancy Project Overview")
st.caption("The Influence of Life Expectancy Determinants: Understanding and Predicting Global Longevity Trends")
st.divider()

# Dataset Introduction 
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

# Load data
df = utils.load_data()

@st.dialog("📊 Explore Dataset Structure & Columns", width="large")
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

# Research Question
st.header("2. ❓ Research Question")
st.header(
    """
    *What are the most significant economic and public health factors (e.g., GDP, immunization rates, mortality) that statistically determine a country's Life Expectancy, and how can countries be segmented into distinct public health groups based on these metrics?*
    """
)
st.divider()

# Analysis Techniques
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

st.divider()

# 4. Data Exploration & Preparation
st.header("4. 🧹 Data Exploration & Preparation")

st.subheader("Data Cleaning")
st.markdown("Checking for missing values in the dataset:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    st.success("✅ No missing values found! The dataset is clean.")
else:
    st.warning("⚠️ Missing values found:")
    st.write(missing_values[missing_values > 0])

st.subheader("Exploratory Visualizations")
tab1, tab2 = st.tabs(["Correlation Heatmap", "Life Expectancy Distribution"])

with tab1:
    st.markdown("### Correlation Matrix")
    st.write("Visualizing how different variables correlate with each other.")
    fig = get_correlation_heatmap(df)
    render_centered_plot(fig)

with tab2:
    st.markdown("### Life Expectancy Distribution")
    fig = get_distribution_plot(df)
    render_centered_plot(fig)

st.divider()

# 5. Analysis and Insights
st.header("5. 📈 Analysis and Insights")

analysis_tab1, analysis_tab2 = st.tabs(["Linear Regression Analysis", "K-Means Clustering Analysis"])

with analysis_tab1:
    st.subheader("Predicting Life Expectancy")
    st.markdown("Select predictor variables to build a Linear Regression model.")
    
    # Feature selection
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    regression_features = [col for col in numeric_columns if col != 'Life_expectancy']
        
    selected_features = st.multiselect(
        "Select Predictors (X)", 
        regression_features, 
        default=['GDP_per_capita', 'Schooling', 'Adult_mortality']
    )
    
    if selected_features:
        # Train model (Cached)
        model, y_pred, X, y = train_regression_model(df, selected_features)
        
        # Metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.4f}", help="How well the model explains the variance")
        col2.metric("Mean Squared Error", f"{mse:.4f}")
        
        # Visualization: Actual vs Predicted
        st.markdown("### Actual vs Predicted Life Expectancy")
        fig = get_regression_plot(y, y_pred)
        render_centered_plot(fig)
        
        # Coefficients
        st.markdown("### Feature Importance (Coefficients)")
        coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        st.dataframe(coef_df, hide_index=True)
    else:
        st.warning("Please select at least one feature.")

with analysis_tab2:
    st.subheader("Clustering Countries")
    st.markdown("Group countries based on selected health and economic indicators.")
    
    # Re-fetch numeric columns for clustering (including Life_expectancy)
    clustering_options = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    cluster_features = st.multiselect(
        "Select Features for Clustering", 
        clustering_options, 
        default=['GDP_per_capita', 'Life_expectancy'],
        key='cluster_features'
    )
    
    k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
    
    if len(cluster_features) >= 2:
        X_cluster = df[cluster_features]
        
        # Train K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    if len(cluster_features) >= 2:
        # Train K-Means (Cached)
        clusters = get_kmeans_clusters(df, cluster_features, k)
        
        df['Cluster'] = clusters
        
        st.markdown(f"### Clusters Visualization (k={k})")
        st.caption("Visualizing the clusters on the first two selected features.")
        
        # Plotting on the first two selected features
        x_axis = cluster_features[0]
        y_axis = cluster_features[1]
        
        fig = get_cluster_plot(df, x_axis, y_axis)
        render_centered_plot(fig)
        
        st.markdown("### Cluster Statistics")
st.divider()

# 6. Conclusions
st.header("6. 💡 Conclusions and Recommendations")

st.markdown(
    """
    ### Key Takeaways
    - **Economic Impact**: Factors like **GDP per capita** and **Schooling** often show a strong positive correlation with Life Expectancy.
    - **Health Factors**: **Adult Mortality** and **HIV Incidents** are critical negative drivers.
    - **Clustering**: Countries naturally group into clusters that reflect their development status, allowing for targeted policy interventions.
    
    ### Recommendations
    1. **Invest in Education**: Increasing schooling years is a powerful lever for long-term health outcomes.
    2. **Strengthen Healthcare Systems**: Reducing adult mortality through better primary care can significantly boost life expectancy.
    3. **Targeted Aid**: Use clustering to identify countries in the "low development" cluster that need specific aid packages focused on basic health infrastructure.
    """
)

# Interactive exploration for conclusion
st.subheader("Explore Specific Country Data")
selected_country = st.selectbox("Select a Country to View Details", df['Country'].unique())
country_data = df[df['Country'] == selected_country]

# Display Economy Status (Static)
is_developed = country_data['Economy_status_Developed'].iloc[0] == 1
status_text = "Developed" if is_developed else "Developing"
st.markdown(f"**Economy Status:** {status_text}")

# Pivot/Set Index to Year and drop redundant columns
country_data = country_data.set_index('Year').sort_index()
country_data = country_data.drop(columns=['Country', 'Region', 'Economy_status_Developed', 'Economy_status_Developing'], errors='ignore')

# Column selection
all_columns = country_data.columns.tolist()

# Initialize session state for applied columns (table view)
if 'applied_columns' not in st.session_state:
    st.session_state.applied_columns = all_columns

# Ensure applied_columns only contains valid columns (fixes KeyError if Cluster was removed/added)
st.session_state.applied_columns = [c for c in st.session_state.applied_columns if c in all_columns]

# Initialize session state for checkbox keys (form view)
for col in all_columns:
    key = f"chk_{col}"
    if key not in st.session_state:
        # Default to True if new column, or respect previous state if exists
        st.session_state[key] = True

with st.popover("Add filter", use_container_width=False):
    st.markdown("### Filter Columns")
    
    # Select All / Clear All buttons
    col1, col2 = st.columns(2)
    if col1.button("Select All", use_container_width=True):
        for col in all_columns:
            st.session_state[f"chk_{col}"] = True
        st.rerun()
        
    if col2.button("Clear All", use_container_width=True):
        for col in all_columns:
            st.session_state[f"chk_{col}"] = False
        st.rerun()

    with st.form("column_filter_form"):
        st.caption("Select columns to display in the table below.")
        
        with st.container(height=300):
            for col in all_columns:
                st.checkbox(col, key=f"chk_{col}")
        
        if st.form_submit_button("Apply Filters"):
            # Update applied columns based on the current state of checkboxes
            selected = [col for col in all_columns if st.session_state.get(f"chk_{col}", False)]
            st.session_state.applied_columns = selected
            st.rerun()

# Use the applied columns from session state
if st.session_state.applied_columns:
    st.dataframe(country_data[st.session_state.applied_columns])
else:
    st.warning("Please select at least one column to display.")
