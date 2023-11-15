import pandas as pd
import streamlit as st
from data import get_clean_data
from utils import dataframe_with_filters, add_logo


st.set_page_config(
    page_title="AIIA - Search",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)
add_logo("img/logo.png", 60)
pd.options.plotting.backend = "plotly"


col_1, col_2 = st.columns([5, 1])
col_1.title("🧭 AI Incidents Explorer")

df, C = get_clean_data()

st.sidebar.link_button(
    "Column descriptions",
    "https://www.aiaaic.org/aiaaic-repository/classifications-and-definitions#h.fyaxuf7wldm7",
    use_container_width=True,
    type="primary",
)

# Display the filtering widgets
df = dataframe_with_filters(
    df,
    on_columns=[C.type, C.country, C.sector, C.technology, C.risks, C.transparency],
    use_sidebar=True,
)

st.data_editor(
    df,
    use_container_width=True,
    height=800,
    hide_index=True,
    disabled=True,
    column_config={
        # C.title: st.column_config.TextColumn(),
        C.type: st.column_config.ListColumn(),
        C.released: st.column_config.NumberColumn(),
        C.occurred: st.column_config.NumberColumn(),
        C.country: st.column_config.ListColumn(),
        C.sector: st.column_config.ListColumn(),
        C.operator: st.column_config.TextColumn(),
        C.developer: st.column_config.TextColumn(),
        C.system_name: st.column_config.TextColumn(),
        C.technology: st.column_config.ListColumn(),
        C.purpose: st.column_config.TextColumn(),
        C.media_trigger: st.column_config.TextColumn(),
        C.risks: st.column_config.ListColumn(),
        C.transparency: st.column_config.ListColumn(),
        C.summary_links: st.column_config.LinkColumn(),
    },
)

col_2.metric("Total incidents displayed", df.index.size)
