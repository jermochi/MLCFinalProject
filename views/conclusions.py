import streamlit as st
import pandas as pd
import utils

def show_conclusions():
    st.header("6. Conclusions and Recommendations")

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

def show_country_exploration(df):
    # Interactive exploration for conclusion
    st.subheader("Explore Specific Country Data")
    
    unique_countries = df['Country'].unique().tolist()
    default_index = 0
    if "Philippines" in unique_countries:
        default_index = unique_countries.index("Philippines")
        
    selected_country = st.selectbox("Select a Country to View Details", unique_countries, index=default_index)
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

    st.markdown(f"**Economy Status:** {status_text}{cluster_text}")

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
        display_df = country_data[st.session_state.applied_columns]
        
        def highlight_changes(data):
            attr_better = 'color: green; font-weight: bold'
            attr_worse = 'color: red; font-weight: bold'
            
            # Create empty style df
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            
            # Define directions
            up_is_good = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 'Polio', 'Diphtheria', 'Hepatitis_B', 'Measles']
            down_is_good = ['Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Incidents_HIV', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Alcohol_consumption']
            
            # Calculate diffs (numeric columns only)
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
            display_df.style.apply(highlight_changes, axis=None).format("{:.2f}"),
            column_config={
                col: st.column_config.Column(
                    help=utils.column_descriptions.get(col, "")
                ) for col in display_df.columns
            }
        )
    else:
        st.warning("Please select at least one column to display.")
