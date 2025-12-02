import streamlit as st
import io
import plots

# --- Caching Wrappers ---
@st.cache_data
def get_correlation_heatmap(df):
    return plots.plot_correlation_heatmap(df)

@st.cache_data
def get_distribution_plot(df):
    return plots.plot_life_expectancy_distribution(df)

@st.cache_data
def get_regression_plot(y, y_pred):
    return plots.plot_actual_vs_predicted(y, y_pred)

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
