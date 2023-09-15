from collections import namedtuple
import streamlit as st

st.set_page_config(page_title="AIIA - Plots", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

df = st.session_state['data']


# A simple pattern for "named tabs",
# this way, one can add tabs on the fly in any order.
TabsNames = namedtuple('_', ["sectors", "released"])
tabs = st.tabs(TabsNames._fields)
tabs = TabsNames(*tabs)


with tabs.sectors:
    st.plotly_chart(df["Sector(s)"].hist(), use_container_width=True)

with tabs.released:
    st.plotly_chart(df.Released.hist(), use_container_width=True)
# st.plotly_chart(df.Occurred.hist(), use_container_width=True)
# st.plotly_chart(df.Type.hist(), use_container_width=True)

# with tabs.page:
    # df.iloc[0]
