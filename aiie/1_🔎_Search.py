import pandas as pd
import streamlit as st

from data import get_clean_data, scrap_incident_description
from utils import dataframe_with_selections, filter_dataframe


st.set_page_config(page_title="AIIA - Search", layout="wide", page_icon="🔎", initial_sidebar_state="expanded")
pd.options.plotting.backend = "plotly"

col_1, col_2 = st.columns([5, 1])
col_1.title(" AI Incidents Explorer")
st.divider()

with st.spinner("Fetchez la data... 🐮") as status:
    df, C = get_clean_data()
    col_2.metric("Total", df.index.size)

# Display the filtering widgets
df = filter_dataframe(df)

# Todo: filtering is not practical when too many categories and one just wants to select a few
# Allow the section of rows (it's hack for now)
df_selected = dataframe_with_selections(df)

for link in df_selected[C.summary_links]:
    st.write(link)
    if link is not None:
        description = scrap_incident_description(link)
        st.write(description)
