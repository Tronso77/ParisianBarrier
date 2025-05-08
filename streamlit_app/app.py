# streamlit_app/app.py
import os, sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import streamlit as st

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")


# 3) Import each dashboard’s entry point
from simulation_dashboard    import show_simulation_dashboard
from pricing_dashboard       import show_pricing_dashboard

# 4) Main navigation sidebar
page = st.sidebar.selectbox(
    "🔹 Select Page",
    ["Simulation", "Pricing"]
)

# 5) Dispatch to the right dashboard
if page == "Simulation":
    show_simulation_dashboard()
else:  # Pricing
    show_pricing_dashboard()
