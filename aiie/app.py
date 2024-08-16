import numpy as np
import pandas as pd
import streamlit as st
from box import Box
from data import get_clean_data
from utils import (
    _df_groupby,
    category_text_filter,
    dataframe_with_filters,
    gen_sankey,
    retain_most_frequent_values,
    scrap_incident_description,
)

st.set_page_config(
    page_title="AI Incidents Explorer",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)

from plotting import interactions, rankings, sankey, timeline, umap

st.html("""
  <style>
    [alt=Logo] {
      height: 10rem;
    }
  </style>
        """)

st.logo(image="img/logo.png", link="http://aiiexp.streamlit.app")
pd.options.plotting.backend = "plotly"


def make_layout():
    layout = Box()

    layout.header = st.columns([5, 1])

    layout.dashboard = st.columns(2, gap="medium")

    layout.data = layout.dashboard[0].container()
    layout.data_config = st.sidebar.container()

    layout.plots = layout.dashboard[1].container()
    layout.plots_config = layout.dashboard[1].container()

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

    df = df[st.session_state.mask]
    with container:
        st.session_state.mask &= dataframe_with_filters(
            df, on_columns=columns_to_filter_on, mask=st.session_state.mask
        )
    df = df[st.session_state.mask]

    with sidebar:
        # Display the filtering widgets
        st.session_state.mask &= category_text_filter(
            df, st.session_state.mask, columns_to_filter_on
        )

    df = df[st.session_state.mask]

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

        show_incident_description(incident[C.summary_links])
    else:
        st.info("Select a row to display more info ", icon="⤴")

    total.metric("Total incidents", df.index.size)


@st.fragment
def show_incident_description(link):
    try:
        incident_description = scrap_incident_description(link)
        st.info(incident_description, icon="📄")
    except:
        st.error("An error occurred. The incident information could not be downloaded.")
    st.page_link(
        link,
        label="Go to the incident page",
        use_container_width=True,
        icon="🌐",
    )


def show_plots(container, sidebar, df, C):
    columns_to_plot = [
        C.country,
        C.type,
        C.sector,
        C.developer,
        C.operator,
        C.technology,
        C.system_name,
        C.risks,
        C.transparency,
        C.media_trigger,
        C.purpose,
    ]

    df = df[st.session_state.mask]

    top_N = 10
    # with sidebar:
    #     top_N = st.select_slider(
    #         "Plot only the most frequent...", [5, 10, 15, 20, 25, 30, 40, 50, "all"], 25
    #     )
    with container:
        plots_tabs = st.tabs(columns_to_plot)
        for i, col in enumerate(columns_to_plot):
            if top_N != "all":
                df_filtered = retain_most_frequent_values(df, col, int(top_N))
            else:
                df_filtered = df
            count_plot = (
                df_filtered[col]
                .value_counts(sort=True, ascending=True)
                .to_frame(name="count")
                .plot(kind="barh")
                .update_layout(showlegend=False)
            )
            plots_tabs[i].plotly_chart(count_plot, use_container_width=True)


def main():
    # get the clean dataset along with the enum mapping of the columns (C)
    df, C = get_clean_data()
    st.session_state.mask = np.full_like(df.index, True, dtype=bool)

    # build the overall layout, creating st.container()s and st.empty()s
    layout = make_layout()
    layout.header[0].title("🧭 AI Incidents Explorer")
    show_raw_data(layout.data, layout.data_config, layout.header[1], df, C)
    show_plots(layout.plots, layout.plots_config, df, C)


pages = {
    "Database": [
        st.Page("about.py", title="About", icon="👋🏼"),
        st.Page(main, title="Search", icon="🔎"),
    ],
    "Plots": [
        # st.Page("plots.py", title="Timeline", icon="📈"),
        st.Page(timeline, title="Timeline", icon="⏳"),
        st.Page(rankings, title="Rankings", icon="🏆"),
        st.Page(sankey, title="Sankey", icon="🤓"),
        st.Page(interactions, title="Interactions", icon="📊"),
        st.Page(umap, title="UMAP", icon="✨"),
    ],
}
pg = st.navigation(pages)
pg.run()
