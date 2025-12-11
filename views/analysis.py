import streamlit as st
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import models
import utils
from countries import mapping
from models import get_kmeans_clusters_with_pca
from . import utils as view_utils

def show_analysis_techniques():
    st.header("3. Selected Analysis Techniques")
    st.markdown(
        """
        We use Linear Regression to quantify the direct relationship between predictors 
        (such as Adult Mortality or GDP) and Life Expectancy, allowing us to see exactly how 
        much a specific factor influences longevity. Simultaneously, we apply K-Means Clustering 
        to detect underlying patterns, grouping countries with similar characteristics to reveal 
        global inequalities and development tiers without bias.
        """
    )

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

    st.markdown(
        """
        This section confirms that our dataset is thoroughly cleaned and free of missing values, 
        ensuring analysis reliability. Through interactive global maps and correlation heatmaps, 
        you can explore the initial geographic trends and statistical relationships that 
        we used for our predictive modeling.
        """
    )

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

def show_model_experimentation(df):
    st.header("5. Model Experimentation and Testing", anchor="5-model-experimentation")

    st.markdown(
        """
        This section serves as a hands-on laboratory exercise on the dataset, allowing you to test 
        hypotheses in real-time. By selecting different predictor variables, such as Schooling 
        or GDP per Capita, you can build custom Linear Regression models to see how well they predict 
        Life Expectancy. Observe the R² Score to judge model accuracy and examine the coefficients 
        to understand whether specific features act as positive drivers or negative barriers to health.
        """
    )

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

def show_analysis_insights(df):
    st.header("6. Analysis and Insights", anchor='6-analysis-and-insights')

    st.markdown(
        """
        The previous sections introduced you to the dataset's features, as well as the common analysis
        techniques used for these datasets. Section 5 also let you experiment with what indicators
        to use along with other parameters of the models to test your hypotheses on what predictors
        influence life expectancy the most.
        """
    )

    st.markdown(
        """
        Now, this section programmatically determines which combination of features predict life expectancy
        the best, along with optimizing the model's parameters.
        """
    )

    analysis_tab1, analysis_tab2 = st.tabs(["Regression", "Clustering"])

    with analysis_tab1:
        st.subheader("Lasso (Regression)")
        st.markdown(
            """
            Simply choosing a number of features and training a regression model is time-consuming and
            often doesn't lead to the best model. Moreover, brute-forcing combinations and selecting 
            models with the highest R² score almost guarantees that the best-performing model is 
            memorizing the data, instead of learning patterns.
            """
        )
        st.markdown(
            """
            This is where Lasso comes in. It performs both variable selection and regularization 
            in order to enhance the prediction accuracy and interpretability of the resulting model. 
            Using this method, irrelevant data features are removed and overfitting is avoided. This
            allows features with weak influence to be clearly identified as 
            the coefficients of less important variables are shrunk toward zero.
            """
        )

        # Get columns and remove non-features
        features = list(df.columns)
        labels = {'Country', 'Region', 'Year', 'Life_expectancy', 'Cluster'}
        features = [x for x in features if x not in labels]

        if st.button('Start Analysis'):
            X = df[features]
            y = df['Life_expectancy']

            # Data scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train Lasso with Cross-Validation (5-fold)
            lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)

            # Get Feature Importance
            coefs = pd.Series(lasso.coef_, index=features)
            print(coefs)

            # Filter useless features (coefficient approx 0)
            threshold = 1e-5
            selected_features = coefs[abs(coefs) > threshold].sort_values(ascending=False, key=abs)
            removed_features = coefs[abs(coefs) < threshold].sort_values(ascending=False)

            # Display Results
            st.subheader("Optimal Features Selected by Lasso")

            col1_1, col1_2 = st.columns([2, 1])

            with col1_1:
                st.bar_chart(selected_features)
                st.caption("Feature weights (Positive = Increases Life Expectancy)")

            with col1_2:
                # Calculate metrics on the full set
                y_pred = lasso.predict(X_scaled)
                st.metric("Overall R² Score", f"{r2_score(y, y_pred):.4f}")
                st.metric("Features Kept", f"{len(selected_features)} / {len(features)}")
                st.metric("Best Alpha", f"{lasso.alpha_:.4f}")

            # Display the actual list
            col2_1, col2_2 = st.columns([2, 1])

            with col2_1:
                st.write("Significant Predictors:", list(selected_features.index))

            with col2_2:
                st.write("Removed Predictors:", list(removed_features.index))

    with analysis_tab2:
        st.subheader("K-Means Clustering with PCA")

        st.markdown(
            """
            Clustering high-dimensional data (datasets with many variables) can be difficult to visualize 
            and prone to noise. To address this, we use **Principal Component Analysis (PCA)** before 
            applying K-Means.

            PCA is a dimensionality reduction technique that transforms our many health and economic 
            indicators into a smaller set of "Principal Components" while retaining the majority of the 
            data's information (variance). 

            By plotting the first two components (**PC1 and PC2**), we can visualize how countries 
            naturally separate on a 2-dimensional plane. The resulting clusters group countries with
            similar development trajectories, allowing us to analyze their specific profiles below.
            """
        )

        with st.expander("Determine Optimal Clusters (Elbow Method)", expanded=True):
            st.write(
                """
                A more systematic way of finding the optimal number of clusters is the Elbow method.
                It helps find the optimal number of clusters by looking for the 
                'elbow' point where the inertia starts decreasing more slowly.
                """
            )

            # Calculate inertia for k=1 to 10
            k_values, inertias = models.calculate_inertia_range(df, features)

            # Plot
            elbow_fig = view_utils.get_elbow_plot(k_values, inertias)
            view_utils.render_centered_plot(elbow_fig)

            st.caption(
                """
                Notice how the change in inertia beyond k=3 is lesser than previous deltas. This indicates
                that adding further clusters yields diminishing returns. Splitting the groups further 
                would result in overfitting (creating artificial distinctions) rather than revealing 
                meaningful insights.
                """
            )

        st.divider()
        st.subheader("Cluster Visualization")

        # User selects K based on the chart above
        number_of_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3)

        st.caption(
            """
            The optimal number of clusters is already selected by default. Observe how increasing the
            number of clusters beyond the optimal number does not group the data points into distinct
            clusters.
            """
        )

        clusters, pca_df = models.get_kmeans_clusters_with_pca(df, features, number_of_clusters)

        # Assign labels
        pca_df['Cluster'] = clusters
        pca_df['Country'] = df['Country']
        df['Cluster_Labels'] = clusters

        # Highlight Country Search
        unique_countries = sorted(df['Country'].unique().tolist())
        highlight_country_analysis = st.selectbox("Highlight Country (Optional)", ["None"] + unique_countries, index=0, key=1)
        if highlight_country_analysis == "None":
            highlight_country = None

        # Plot PCA
        fig = view_utils.get_cluster_plot(pca_df, 'PC1', 'PC2', highlight_country_analysis)
        view_utils.render_centered_plot(fig)

        st.subheader("Cluster Profiles")
        stats_to_show = st.multiselect(
            "Select Predictors to Compare",
            features,
            default=['GDP_per_capita', 'Schooling']
        )

        if stats_to_show:
            profile = df.groupby('Cluster_Labels')[stats_to_show].mean()
            st.dataframe(
                profile.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))

    st.divider()