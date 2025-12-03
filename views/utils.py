import streamlit as st
import plots

# --- Caching Wrappers ---
@st.cache_data
def get_correlation_heatmap(df):
    return plots.plot_correlation_heatmap(df)

@st.cache_data
def get_distribution_plot(df):
    return plots.plot_life_expectancy_distribution(df)

@st.cache_data
def get_regression_plot(y, y_pred, country_names=None, highlight_country=None):
    return plots.plot_actual_vs_predicted(y, y_pred, country_names, highlight_country)

@st.cache_data
def get_cluster_plot(df, x_axis, y_axis, highlight_country=None):
    return plots.plot_clusters(df, x_axis, y_axis, highlight_country)

def render_centered_plot(fig):
    """Renders a plotly figure centered."""
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
