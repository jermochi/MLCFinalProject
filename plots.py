import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=False, aspect="equal", color_continuous_scale='RdBu_r', origin='lower', height=800)
    fig.update_layout(title="Correlation Heatmap", xaxis_fixedrange=True, yaxis_fixedrange=True, dragmode=False)
    return fig

def plot_life_expectancy_distribution(df):
    fig = px.histogram(df, x='Life_expectancy', nbins=30, title="Distribution of Life Expectancy", height=800)
    fig.update_layout(bargap=0.1, xaxis_fixedrange=True, yaxis_fixedrange=True, dragmode=False)
    return fig

def plot_actual_vs_predicted(y_actual, y_predicted, country_names=None, highlight_country=None):
    results_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
    
    hover_data = None
    if country_names is not None:
        results_df['Country'] = country_names.values
        hover_data = ['Country']

    fig = px.scatter(results_df, x='Actual', y='Predicted', opacity=0.5, title="Actual vs Predicted Life Expectancy", height=800, hover_data=hover_data, render_mode='svg')
    
    # Add diagonal line
    fig.add_shape(
        type="line",
        x0=y_actual.min(), y0=y_actual.min(),
        x1=y_actual.max(), y1=y_actual.max(),
        line=dict(color="Red", dash="dash"),
    )

    # Highlight specific country
    if highlight_country and country_names is not None:
        highlight_df = results_df[results_df['Country'] == highlight_country]
        if not highlight_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlight_df['Actual'],
                    y=highlight_df['Predicted'],
                    mode='markers',
                    marker=dict(color='yellow', size=10, symbol='circle', opacity=1.0),
                    name=highlight_country,
                    hoverinfo='skip' # Already covered by main trace or we can make it custom
                )
            )
            # Force highlight trace to be last (drawn on top)
            fig.data = tuple([t for t in fig.data if t.name != highlight_country] + [t for t in fig.data if t.name == highlight_country])

    fig.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True, dragmode=False)
    return fig

def plot_clusters(df, x_axis, y_axis, highlight_country=None):
    # Ensure Cluster is treated as categorical for discrete colors
    df['Cluster_Label'] = df['Cluster'].astype(str)
    fig = px.scatter(
        df, 
        x=x_axis, 
        y=y_axis, 
        color='Cluster_Label', 
        title=f"Clustering: {x_axis} vs {y_axis}",
        hover_data=['Country'],
        height=800,
        render_mode='svg'
    )

    # Highlight specific country
    if highlight_country:
        highlight_df = df[df['Country'] == highlight_country]
        if not highlight_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlight_df[x_axis],
                    y=highlight_df[y_axis],
                    mode='markers',
                    marker=dict(color='yellow', size=10, symbol='circle', opacity=1.0),
                    name=highlight_country,
                    hoverinfo='skip'
                )
            )
            # Force highlight trace to be last (drawn on top)
            fig.data = tuple([t for t in fig.data if t.name != highlight_country] + [t for t in fig.data if t.name == highlight_country])

    fig.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True, dragmode=False)
    return fig
