import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import models
from . import utils as view_utils

def show_analysis_techniques():
    st.header("3. Selected Analysis Techniques")
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

def show_data_exploration(df):
    st.header("4. Data Exploration & Preparation")

    st.subheader("Data Cleaning")
    st.markdown("Checking for missing values in the dataset:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("No missing values found! The dataset is clean.")
    else:
        st.warning("Missing values found:")
        st.write(missing_values[missing_values > 0])

    st.subheader("Exploratory Visualizations")
    tab1, tab2 = st.tabs(["Correlation Heatmap", "Life Expectancy Distribution"])

    with tab1:
        st.markdown("### Correlation Matrix")
        st.write("Visualizing how different variables correlate with each other.")
        fig = view_utils.get_correlation_heatmap(df)
        view_utils.render_centered_plot(fig)

    with tab2:
        st.markdown("### Life Expectancy Distribution")
        fig = view_utils.get_distribution_plot(df)
        view_utils.render_centered_plot(fig)

    st.divider()

def show_analysis_insights(df):
    st.header("5. Analysis and Insights")

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
            model, y_pred, X, y = models.train_regression_model(df, selected_features)
            
            # Metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            col1, col2 = st.columns(2)
            col1.metric("R² Score", f"{r2:.4f}", help="How well the model explains the variance")
            col2.metric("Mean Squared Error", f"{mse:.4f}")
            
            # Visualization: Actual vs Predicted
            st.markdown("### Actual vs Predicted Life Expectancy")
            fig = view_utils.get_regression_plot(y, y_pred)
            view_utils.render_centered_plot(fig)
            
            # Coefficients
            st.markdown("### Feature Importance (Coefficients)")
            st.caption(
                """
                **Interpretation:**
                *   **Positive Coefficient (+):** As this feature increases, Life Expectancy tends to **increase**.
                *   **Negative Coefficient (-):** As this feature increases, Life Expectancy tends to **decrease**.
                *   **Magnitude:** Larger absolute values indicate a stronger influence on the prediction.
                """
            )
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
            # Train K-Means (Cached)
            clusters = models.get_kmeans_clusters(df, cluster_features, k)
            
            df['Cluster'] = clusters
            
            st.markdown(f"### Clusters Visualization (k={k})")
            st.caption("Visualizing the clusters on the first two selected features.")
            
            # Plotting on the first two selected features
            x_axis = cluster_features[0]
            y_axis = cluster_features[1]
            
            fig = view_utils.get_cluster_plot(df, x_axis, y_axis)
            view_utils.render_centered_plot(fig)
            
            st.markdown("### Cluster Statistics")
            st.caption("The table below shows the **average (mean) value** of each feature for every cluster. This helps identify the characteristics that define each group (e.g., 'Cluster 0 has high GDP but low Life Expectancy').")
            cluster_stats = df.groupby('Cluster')[cluster_features].mean()
            st.dataframe(cluster_stats)
        else:
            st.warning("Please select at least two features for clustering.")
    st.divider()
