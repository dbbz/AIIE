import numpy as np
import pandas as pd
import streamlit as st
from data import get_clean_data
from utils import (
    dataframe_with_filters,
    github_repo_url,
    retain_most_frequent_values,
    _df_groupby,
    gen_sankey,
    plot_counts,
    add_logo,
)

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
table_height = 800

# columns_to_plot = list(map(str, C))  # get all the column names
available_columns_to_plot = [
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


with st.sidebar.expander("Plotting", expanded=True):
    default_value = query_parameters.get("Columns", C.country)
    columns_to_plot = st.multiselect(
        "Columns",
        available_columns_to_plot,
        default_value,
        label_visibility="collapsed",
    )
    # top_N = st.number_input("Number of top values to show", 0, 20, 10)
    top_N = st.select_slider(
        "Show most frequent...", [5, 10, 15, 20, 25, 30, 40, 50, "all"], 25
    )

    # if st.sidebar.button("Save"):
    #     st.experimental_set_query_params(plotted=columns_to_plot)

    enable_sankey = st.toggle("Sankey plot", False)


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

if columns_to_plot or enable_sankey:
    table_height = 600
    if enable_sankey:
        tabs = st.tabs(columns_to_plot + ["Sankey plot"])
    else:
        tabs = st.tabs(columns_to_plot)
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
        tabs[i].plotly_chart(count_plot, use_container_width=True)
    if enable_sankey:
        with tabs[-1]:
            sankey_vars = st.multiselect(
                "Choose at least two columns to plot",
                available_columns_to_plot,
                default=columns_to_plot[:2],
                max_selections=4,
                help="💡 Use the text filters for better plots.",
            )

            if sankey_vars:
                sankey_cols = st.columns(len(sankey_vars))
            text_filters = {}
            for i, col in enumerate(sankey_vars):
                text_filters[col] = sankey_cols[i].text_input(
                    "Text filter on " + col,
                    key="text_" + col,
                    help="Case-insensitive text filtering.",
                )

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
                    None,
                    # " - ".join(sankey_vars),
                )
                st.plotly_chart(fig, use_container_width=True)


st.data_editor(
    df,
    use_container_width=True,
    height=table_height,
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
