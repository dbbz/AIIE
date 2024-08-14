import numpy as np
import pandas as pd
import streamlit as st
from box import Box

from data import get_clean_data
from utils import (
    category_text_filter,
    dataframe_with_filters,
    scrap_incident_description,
)

st.set_page_config(
    page_title="AI Incidents Explorer",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)


st.logo(image="img/logo.png", link="http://aiiexp.streamlit.app")
pd.options.plotting.backend = "plotly"


def make_layout():
    layout = Box()
    layout.sidebar = {}

    with st.sidebar:
        layout.sidebar.data_config = st.container()
        layout.sidebar.plotting_config = st.container()

    layout.header = st.columns([5, 1])
    layout.data = st.container()

    return layout


def show_raw_data(container, sidebar, total, df, C):
    columns_to_filter_on = [
        # C.occurred,
        # C.released,
        C.type,
        C.country,
        C.sector,
        C.technology,
        C.risks,
        C.transparency,
        C.media_trigger,
    ]
    mask = np.full_like(df.index, True, dtype=bool)

    with sidebar:
        # Display the filtering widgets
        mask = category_text_filter(df, mask, columns_to_filter_on)

    df = df[mask]
    with container:
        mask = dataframe_with_filters(df, on_columns=columns_to_filter_on, mask=mask)
    df = df[mask]

    with container:
        selected_row = st.dataframe(
            df,
            use_container_width=True,
            # height=400,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
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

    if selected_row.selection.rows:
        incident_index = selected_row.selection.rows[0]
        incident = df.iloc[incident_index]

        incident_description = scrap_incident_description(incident[C.summary_links])

        st.info(incident_description, icon="📄")
        st.page_link(
            incident[C.summary_links],
            label="Go to the incident page",
            use_container_width=True,
            icon="🌐",
        )
    else:
        st.info("Select a row to display more info ", icon="⤴")

    total.metric("Total incidents", df.index.size)


def main():
    # get the clean dataset along with the enum mapping of the columns (C)
    df, C = get_clean_data()

    # build the overall layout, creating st.container()s and st.empty()s
    layout = make_layout()

    layout.header[0].title("🧭 AI Incidents Explorer")

    show_raw_data(layout.data, layout.sidebar.data_config, layout.header[1], df, C)


from plotting import plot_timeline, plot_incidents_ranking, plot_sankey

pages = {
    "Search": [st.Page(main, title="Search", icon="🔎")],
    "Plots": [
        # st.Page("plots.py", title="Timeline", icon="📈"),
        st.Page(plot_timeline, title="Timeline", icon="⏳"),
        st.Page(plot_incidents_ranking, title="Rankings", icon="🏆"),
        st.Page(plot_sankey, title="Sankey", icon="🤓"),
        # st.Page(main, title="Charts", icon="📊"),
    ],
    "About": [st.Page("about.py", title="About", icon="👋🏼")],
}
pg = st.navigation(pages)
pg.run()
