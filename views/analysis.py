import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import models
import utils
from countries import mapping
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


@st.dialog("Country Details", width="large")
def show_country_details(df, selected_country):
    country_data = df[df['Country'] == selected_country]

    # Display Economy Status (Static)
    is_developed = country_data['Economy_status_Developed'].iloc[0] == 1
    status_text = "Developed" if is_developed else "Developing"

    # Handle Cluster Display
    cluster_text = ""
    if 'Cluster' in country_data.columns:
        # Get the most frequent cluster (mode)
        cluster_mode = country_data['Cluster'].mode()[0]
        cluster_text = f" | **Cluster:** {cluster_mode}"
        # Drop Cluster from data to be shown in table
        country_data = country_data.drop(columns=['Cluster'])

    st.markdown(f"### {selected_country}")
    st.markdown(f"**Economy Status:** {status_text}{cluster_text}")

    # Pivot/Set Index to Year and drop redundant columns
    country_data = country_data.set_index('Year').sort_index()
    country_data = country_data.drop(columns=['Country', 'Region', 'Economy_status_Developed', 'Economy_status_Developing'], errors='ignore')

    # Column selection
    all_columns = country_data.columns.tolist()

    # Initialize session state for applied columns (table view)
    # We use a unique key prefix for this dialog to avoid conflicts
    ss_key_applied = f"applied_columns_{selected_country}"
    if ss_key_applied not in st.session_state:
        st.session_state[ss_key_applied] = all_columns

    # Ensure applied_columns only contains valid columns
    st.session_state[ss_key_applied] = [c for c in st.session_state[ss_key_applied] if c in all_columns]

    # Initialize session state for checkbox keys (form view)
    for col in all_columns:
        key = f"chk_{col}_{selected_country}"
        if key not in st.session_state:
            st.session_state[key] = True

    with st.popover("Add filter", use_container_width=False):
        st.markdown("### Filter Columns")
        
        col1, col2 = st.columns(2)
        if col1.button("Select All", key=f"btn_sel_all_{selected_country}", use_container_width=True):
            for col in all_columns:
                st.session_state[f"chk_{col}_{selected_country}"] = True
            st.rerun()
            
        if col2.button("Clear All", key=f"btn_clr_all_{selected_country}", use_container_width=True):
            for col in all_columns:
                st.session_state[f"chk_{col}_{selected_country}"] = False
            st.rerun()

        with st.form(f"column_filter_form_{selected_country}"):
            st.caption("Select columns to display in the table below.")
            
            with st.container(height=300):
                for col in all_columns:
                    st.checkbox(col, key=f"chk_{col}_{selected_country}")
            
            if st.form_submit_button("Apply Filters"):
                selected = [col for col in all_columns if st.session_state.get(f"chk_{col}_{selected_country}", False)]
                st.session_state[ss_key_applied] = selected
                st.rerun()

    if st.session_state[ss_key_applied]:
        display_df = country_data[st.session_state[ss_key_applied]]
        
        def highlight_changes(data):
            attr_better = 'color: green; font-weight: bold'
            attr_worse = 'color: red; font-weight: bold'
            
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            
            up_is_good = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Polio', 'Diphtheria', 'Hepatitis_B', 'Measles']
            down_is_good = ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Incidents_HIV', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Alcohol_consumption']
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            diffs = data[numeric_cols].diff()
            
            for col in numeric_cols:
                if col in up_is_good:
                    style_df.loc[diffs[col] > 0, col] = attr_better
                    style_df.loc[diffs[col] < 0, col] = attr_worse
                elif col in down_is_good:
                    style_df.loc[diffs[col] > 0, col] = attr_worse
                    style_df.loc[diffs[col] < 0, col] = attr_better
                    
            return style_df

        st.dataframe(
            display_df.style.apply(highlight_changes, axis=None).format("{:.2f}", subset=display_df.select_dtypes(include=['number']).columns),
            column_config={
                col: st.column_config.Column(
                    help=utils.column_descriptions.get(col, "")
                ) for col in display_df.columns
            }
        )
    else:
        st.warning("Please select at least one column to display.")

def show_data_exploration(df):
    st.header("4. Data Exploration & Preparation", anchor="4-data-exploration-preparation")

    st.subheader("Data Cleaning")
    st.markdown("Checking for missing values in the dataset:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("No missing values found! The dataset is clean.")
    else:
        st.warning("Missing values found:")
        st.write(missing_values[missing_values > 0])

    st.subheader("Exploratory Visualizations")
    tab1, tab2, tab3 = st.tabs(["Global Interactive Map", "Correlation Heatmap", "Life Expectancy Distribution"])

    with tab1:
        st.markdown("### Global Life Expectancy Map")
        st.markdown("Hover over a country to see details. **Click on a country to open a detailed exploration popup.**")
        
        fig = view_utils.get_interactive_map(df)

        # Display map and capture selection
        event = st.plotly_chart(
            fig, 
            on_select="rerun", 
            selection_mode="points", 
            use_container_width=True,
            config={'scrollZoom': True, 'displayModeBar': True}
        )

        if event and "selection" in event and event["selection"]["points"]:
            # Get selected country ISO Code from map
            point = event["selection"]["points"][0]
            
            selected_iso = None
            if "location" in point:
                selected_iso = point["location"]
            
            if selected_iso:
                # get map (cached if possible, but fast enough here)
                unique_countries = df['Country'].unique()
                iso_map = {mapping.get_iso3(c): c for c in unique_countries}
                
                selected_country_name = iso_map.get(selected_iso)
                
                if selected_country_name:
                    show_country_details(df, selected_country_name)
                else:
                    st.error(f"Could not find data for country code: {selected_iso}")

    with tab2:
        st.markdown("### Correlation Matrix")
        st.write("Visualizing how different variables correlate with each other.")
        fig = view_utils.get_correlation_heatmap(df)
        view_utils.render_centered_plot(fig)

    with tab3:
        st.markdown("### Life Expectancy Distribution")
        fig = view_utils.get_distribution_plot(df)
        view_utils.render_centered_plot(fig)

    st.divider()

def show_analysis_insights(df):
    st.header("5. Analysis and Insights")
    # Highlight Country Search
    unique_countries = sorted(df['Country'].unique().tolist())
    highlight_country = st.selectbox("Highlight Country (Optional)", ["None"] + unique_countries, index=0)
    if highlight_country == "None":
        highlight_country = None

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
            fig = view_utils.get_regression_plot(y, y_pred, df['Country'], highlight_country)
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
            
            fig = view_utils.get_cluster_plot(df, x_axis, y_axis, highlight_country)
            view_utils.render_centered_plot(fig)
            
            st.markdown("### Cluster Statistics")
            st.caption("The table below shows the **average (mean) value** of each feature for every cluster. This helps identify the characteristics that define each group (e.g., 'Cluster 0 has high GDP but low Life Expectancy').")
            cluster_stats = df.groupby('Cluster')[cluster_features].mean()
            st.dataframe(cluster_stats)
        else:
            st.warning("Please select at least two features for clustering.")
    st.divider()
