import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    return fig

def plot_life_expectancy_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df['Life_expectancy'], kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribution of Life Expectancy")
    return fig

def plot_actual_vs_predicted(y_actual, y_predicted):
    results_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(x='Actual', y='Predicted', data=results_df, alpha=0.5, ax=ax)
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2) # Diagonal line
    ax.set_xlabel("Actual Life Expectancy")
    ax.set_ylabel("Predicted Life Expectancy")
    return fig

def plot_clusters(df, x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Cluster', palette='viridis', ax=ax)
    ax.set_title(f"Clustering: {x_axis} vs {y_axis}")
    return fig
