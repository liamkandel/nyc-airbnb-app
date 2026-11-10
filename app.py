"""NYC Airbnb Price Explorer & Estimator — Streamlit entry point.

Run locally:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="NYC Airbnb Price Explorer",
    page_icon=":material/apartment:",
    layout="wide",
)

pages = [
    st.Page("views/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/explore.py", title="Explore listings", icon=":material/map:"),
    st.Page("views/neighborhoods.py", title="Compare neighborhoods", icon=":material/bar_chart:"),
    st.Page("views/estimator.py", title="Price estimator", icon=":material/calculate:"),
    st.Page("views/performance.py", title="Model performance", icon=":material/insights:"),
    st.Page("views/api_docs.py", title="REST API", icon=":material/api:"),
]
st.navigation(pages).run()
