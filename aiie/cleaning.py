import shelve
from itertools import chain

import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup
from data import get_clean_data, scrap_incident_description
from utils import named_tabs, dataframe_with_filters

import plotly.express as px


# Password protect to avoid annotation vandalism
def check_password():
    """Returns `True` if the user had the correct password."""

    import hmac

    if "debug_mode" in st.secrets and st.secrets["debug_mode"]:
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

if st.sidebar.button("Re-download repository"):
    st.cache_data.clear()
df, C = get_clean_data(raw=True)

# tabs = named_tabs("Countries", "Other")


# with tabs.Countries:

columns = [col for col in df.columns if col not in (C.title, C.summary_links)]

col = st.selectbox("Select a column to investigate", columns)

counts = df[col].value_counts(dropna=False).to_frame().reset_index()
mask = dataframe_with_filters(
    counts, on_columns=counts.columns, text="Filter on specific categories"
)
counts = counts[mask]

st.dataframe(
    counts,
    use_container_width=True,
    hide_index=True,
    column_config={"count": st.column_config.Column(width=200)},
)

st.plotly_chart(counts.set_index(col).plot.barh())

# with tabs.Other:
#     unique_values = df[C.country].unique()
#     unique_values = (
#         pd.Series(unique_values, dtype="string")
#         .str.replace(";", ",")
#         .str.strip()
#         .str.strip(",")
#         .str.split(",")
#         .dropna()
#     )
#     unique_values = pd.Series(sorted(list(chain.from_iterable(unique_values))))
#     unique_values = unique_values.value_counts().sort_index()

#     st.dataframe(unique_values, width=500, height=700)
