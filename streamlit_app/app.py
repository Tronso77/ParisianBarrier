# streamlit_app/app.py
import os
import sys
import streamlit as st

# ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import dashboard entrypoints
from simulation_dashboard import show_simulation_dashboard
from pricing_dashboard    import show_pricing
#from dashboards.structuring import show_structuring

st.set_page_config(
    page_title="Monte Carlo & Structuring App",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Monte Carlo Simulator")
page = st.sidebar.radio(
    "Navigate to",
    options=["Simulation", "Pricing", "Structuring"]
)

# Dispatch
if page == "Simulation":
    show_simulation_dashboard()
elif page == "Pricing":
    show_pricing()

