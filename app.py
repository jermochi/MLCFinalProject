import streamlit as st
import utils
import views

# Page Configuration
st.set_page_config(page_title="MLC Final Project", page_icon="🌍", layout="wide")

# Sidebar Navigation
views.show_sidebar()

# Project Overview
views.show_overview()

# Load data
df = utils.load_data()

# Dataset Introduction
views.show_dataset_introduction(df)

# Research Question
views.show_research_question()

# Analysis Techniques
views.show_analysis_techniques()

# Data Exploration & Preparation
views.show_data_exploration(df)

# Analysis and Insights
views.show_analysis_insights(df)

# Conclusions
views.show_conclusions()

# Interactive exploration for conclusion
views.show_country_exploration(df)
