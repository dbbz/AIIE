import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from data import get_clean_data
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from utils import (
    _df_groupby,
    dataframe_with_filters,
    gen_sankey,
    github_repo_url,
    plot_counts,
    retain_most_frequent_values,
)

st.logo(image="img/logo.png", link="http://aiiexp.streamlit.app")
st.sidebar.info(
    f"""
    Want to see another plot in particular?
    [Let me know by opening a GitHub issue!]({github_repo_url})
    """,
    icon="👾",
)

df, C = get_clean_data()


def plot_timeline():
    top_N = st.sidebar.number_input("Number of top values to show", 1, 20, 10)

    st.markdown("#### How is the evolution over the years?")

    data = (
        df[C.occurred].value_counts().rename_axis("Year").reset_index(name="Incidents")
    )
    data = data.sort_values(by="Year")  # Ensure data is sorted by year

    fig = px.area(data, x="Year", y="Incidents", line_shape="spline")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### What about emerging actors?")

    columns_for_timeline = [C.developer, C.risks, C.sector]
    tabs = st.tabs(["AI developers", "Risks", "Sectors"])
    for col_to_plot, tab in zip(columns_for_timeline, tabs):
        df_filtered = retain_most_frequent_values(df, col_to_plot, int(top_N))
        with tab:
            df_gr = (
                df_filtered.groupby([C.occurred, col_to_plot])
                .size()
                .to_frame(name="counts")
                .reset_index()
                .sort_values(by="counts", ascending=False)
            )

            col_1, col_2 = st.columns(2)

            with col_1:
                st.markdown("#### Counts")
                fig_counts = px.area(
                    df_gr,
                    x=C.occurred,
                    y="counts",
                    color=col_to_plot,
                    hover_name=col_to_plot,
                    log_y=False,
                    line_shape="spline",
                )
                fig_counts.update_layout(showlegend=False)
                st.plotly_chart(fig_counts, use_container_width=True)

            with col_2:
                st.markdown("#### Fractions")
                fig_fractions = px.area(
                    df_gr,
                    x=C.occurred,
                    y="counts",
                    color=col_to_plot,
                    hover_name=col_to_plot,
                    groupnorm="percent",
                    line_shape="spline",
                )
                st.plotly_chart(fig_fractions, use_container_width=True)


def plot_incidents_ranking():
    top_N = st.sidebar.number_input("Number of top values to show", 1, 20, 10)

    tabs = st.tabs(["by AI developers", "by countries", "by sectors", "by risk"])
    with tabs[0]:
        question = f"#### What are the top {top_N} AI developers involved in incidents?"
        st.markdown(question)
        plot_counts(df, C.developer, top_N)

    with tabs[1]:
        question = f"#### What are the top {top_N} countries related to AI incidents?"
        st.markdown(question)
        plot_counts(df, C.country, top_N)

    with tabs[2]:
        question = f"#### What are the top {top_N} sectors involving AI incidents?"
        st.markdown(question)
        plot_counts(df, C.sector, top_N)

    with tabs[3]:
        question = f"#### What are the top {top_N} risks?"
        st.markdown(question)
        plot_counts(df, C.risks, top_N)


def plot_sankey():
    columns_to_plot = [
        C.sector,
        C.type,
        C.technology,
        C.risks,
        C.country,
        C.operator,
        C.transparency,
    ]

    sankey_vars = st.multiselect(
        "Choose at least two columns to plot",
        columns_to_plot,
        default=columns_to_plot[:2],
        max_selections=4,
        help="💡 Use the left sidebar to add filters. ",
    )
    st.info(
        "Use the text filters on the sidebar for more precision and clarify.",
        icon="💡",
    )
    if sankey_vars:
        st.sidebar.markdown("#### Sankey plot controls")
    text_filters = {}
    with st.sidebar.expander("Sankey text filters", expanded=True):
        for col in sankey_vars:
            text_filters[col] = st.text_input(
                col,
                key="text_" + col,
                help="Case-insensitive text filtering.",
            )

    # with st.sidebar.expander("Other filters", expanded=False):
    #     for col in columns_to_plot:
    #         if col not in sankey_vars:
    #             text_filters[col] = st.text_input(
    #                 col,
    #                 key="text_" + col,
    #             )

    if len(sankey_vars) == 1:
        st.warning("Select a second column to plot.", icon="⚠️")

    mask = np.full_like(df.index, True, dtype=bool)
    for col, filered_text in text_filters.items():
        if filered_text.strip():
            mask = mask & df[col].str.lower().str.contains(filered_text.lower())

    if len(sankey_vars) > 1:
        df_mask = df[mask]
        df_sankey = _df_groupby(df_mask, sankey_vars)
        fig = gen_sankey(
            df_sankey,
            sankey_vars,
            "counts",
            " - ".join(sankey_vars),
        )
        st.plotly_chart(fig, use_container_width=True)
