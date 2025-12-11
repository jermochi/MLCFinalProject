import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@st.cache_data
def train_regression_model(df, selected_features):
    X = df[selected_features]
    y = df['Life_expectancy']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    return model, y_pred, X, y

@st.cache_data
def get_kmeans_clusters(df, cluster_features, k):
    X_cluster = df[cluster_features]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    return clusters

@st.cache_data
def get_kmeans_clusters_with_pca(df, cluster_features, k):
    X = df[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    return clusters, pca_df

@st.cache_data
def calculate_inertia_range(df, features, k_min=1, k_max=10):
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    k_range = range(k_min, k_max + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    return list(k_range), inertias
