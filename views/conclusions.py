import streamlit as st


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


