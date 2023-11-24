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
add_logo("img/logo.png", 90)
pd.options.plotting.backend = "plotly"

query_parameters = st.experimental_get_query_params()

col_1, col_2 = st.columns([5, 1])
col_1.title("🧭 AI Incidents Explorer")

df, C = get_clean_data()

# columns_to_plot = list(map(str, C))  # get all the column names
columns_to_plot = [
    C.country,
    C.type,
    C.sector,
    C.developer,
    C.technology,
    C.risks,
    C.transparency,
    C.media_trigger,
    C.developer,
    C.operator,
    C.purpose,
]

default_value = query_parameters.get("plotted", C.country)
columns_to_plot = st.sidebar.multiselect("Plotted", columns_to_plot, default_value)

if st.sidebar.button("Save"):
    st.experimental_set_query_params(plotted=columns_to_plot)

# Display the filtering widgets
df = dataframe_with_filters(
    df,
    on_columns=[
        # C.occurred,
        # C.released,
        C.type,
        C.country,
        C.sector,
        C.technology,
        C.risks,
        C.transparency,
        C.media_trigger,
    ],
    use_sidebar=True,
)

if columns_to_plot:
    tabs = st.tabs(columns_to_plot)
    for i, col in enumerate(columns_to_plot):
        count_plot = (
            df[col]
            .value_counts(sort=True, ascending=True)
            .to_frame(name="count")
            .plot(kind="barh")
            .update_layout(showlegend=False)
        )
        tabs[i].plotly_chart(count_plot, use_container_width=True)

st.data_editor(
    df,
    use_container_width=True,
    # height=400,
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
