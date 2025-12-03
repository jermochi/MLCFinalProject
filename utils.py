import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    import os
    if os.path.exists('data/life-exp-data.parquet'):
        return pd.read_parquet('data/life-exp-data.parquet')
    return pd.read_csv('data/life-exp-data.csv')

column_descriptions = {
    "Country": "List of the 179 countries",
    "Region": "179 countries are distributed in 9 regions. E.g. Africa, Asia, Oceania, European Union, Rest of Europe",
    "Year": "Years observed from 2000 to 2015",
    "Infant_deaths": "Represents infant deaths per 1000 population",
    "Under_five_deaths": "Represents deaths of children under five years old per 1000 population",
    "Adult_mortality": "Represents deaths of adults per 1000 population",
    "Alcohol_consumption": "Represents alcohol consumption that is recorded in liters of pure alcohol per capita with 15+ years old",
    "Hepatitis_B": "Represents % of coverage of Hepatitis B (HepB3) immunization among 1-year-olds.",
    "Measles": "Represents % of coverage of Measles containing vaccine first dose (MCV1) immunization among 1-year-olds.",
    "BMI": "BMI is a measure of nutritional status in adults. It is defined as a person's weight in kilograms divided by the square of their height in meters (kg/m2).",
    "Polio": "Represents % of coverage of Polio (Pol3) immunization among 1-year-olds.",
    "Diphtheria": "Represents % of coverage of Diphtheria tetanus toxoid and pertussis (DTP3) immunization among 1-year-olds.",
    "Incidents_HIV": "Incidents of HIV per 1000 population aged 15-49",
    "GDP_per_capita": "GDP per capita in current USD",
    "Population_mln": "Total population in millions",
    "Thinness_ten_nineteen_years": "Prevalence of thinness among adolescents aged 10-19 years. BMI < -2 standard deviations below the median.",
    "Thinness_five_nine_years": "Prevalence of thinness among children aged 5-9 years. BMI < -2 standard deviations below the median.",
    "Schooling": "Average years that people aged 25+ spent in formal education",
    "Life_expectancy": "Average life expectancy of both genders in different years from 2010 to 2015",
    "Economy_status_Developed": "Developed country",
    "Economy_status_Developing": "Developing county"
}

def get_column_info(df):
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.astype(str).values,
        'Sample Value': df.iloc[0].values
    })
    col_info['Description'] = col_info['Column Name'].map(column_descriptions).fillna(" - ")
    return col_info
