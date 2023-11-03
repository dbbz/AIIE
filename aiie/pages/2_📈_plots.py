import streamlit as st
from data import get_clean_data
from utils import named_tabs
import plotly.express as px
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="AIIA - Plots",
    layout="centered",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

df, C = get_clean_data()

tabs = named_tabs("Rankings", "Timing", "Sankey")

columns_to_plot = [C.developer, C.operational, C.country, C.sector]

with tabs.Rankings:
    # top_N = st.sidebar.number_input("Number of ", 3, 20, 15, 1)
    top_N = 15
    col = st.selectbox("Select a variable", columns_to_plot)
    question = f"Who are the top {top_N} {col} involved in incidents?"
    # with st.expander(question, expanded=True):
    st.subheader(question)
    df_counts = df[col].value_counts(sort=True, ascending=True).to_frame().iloc[-top_N:]
    st.plotly_chart(
        df_counts.plot(kind="barh").update_layout(showlegend=False),
        use_container_width=True,
    )


@st.cache_data
def _df_groupby(df, cols):
    return df.groupby(cols).size().to_frame(name="counts").reset_index()


def gen_sankey(df, cat_cols=[], value_cols="", title="Sankey Diagram"):
    # maximum of 6 value cols -> 6 colors
    color_palette = ["#4B8BBE", "#306998", "#FFE873", "#FFD43B", "#646464"]
    label_list = []
    color_num_list = []
    for cat_col in cat_cols:
        label_list_temp = list(set(df[cat_col].values))
        color_num_list.append(len(label_list_temp))
        label_list = label_list + label_list_temp

    # remove duplicates from label_list
    label_list = list(dict.fromkeys(label_list))

    # define colors based on number of levels
    color_list = []
    for idx, color_num in enumerate(color_num_list):
        color_list = color_list + [color_palette[idx]] * color_num

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            source_target_df = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            source_target_df.columns = ["source", "target", "count"]
        else:
            temp_df = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            temp_df.columns = ["source", "target", "count"]
            source_target_df = pd.concat([source_target_df, temp_df])
        source_target_df = (
            source_target_df.groupby(["source", "target"])
            .agg({"count": "sum"})
            .reset_index()
        )

    # add index for source-target pair
    source_target_df["sourceID"] = source_target_df["source"].apply(
        lambda x: label_list.index(x)
    )
    source_target_df["targetID"] = source_target_df["target"].apply(
        lambda x: label_list.index(x)
    )

    # creating the sankey diagram
    data = dict(
        type="sankey",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label_list,
            # color=color_list,
        ),
        link=dict(
            source=source_target_df["sourceID"],
            target=source_target_df["targetID"],
            value=source_target_df["count"],
        ),
    )

    layout = dict(title=title, font=dict(size=10), height=1200)

    fig = dict(data=[data], layout=layout)
    return fig


with tabs.Sankey:
    sankey_vars = st.multiselect(
        "Choose at least two columns to plot",
        columns_to_plot,
        default=columns_to_plot[:2],
    )

    text_filters = {}
    with st.sidebar.expander("Columns filters", expanded=True):
        for col in sankey_vars:
            text_filters[col] = st.text_input(
                col,
                key="text_" + col,
            )

    with st.sidebar.expander("Other filters", expanded=False):
        for col in columns_to_plot:
            if col not in sankey_vars:
                text_filters[col] = st.text_input(
                    col,
                    key="text_" + col,
                )

    if len(sankey_vars) == 1:
        st.info("Select a second column to plot.")

    mask = np.full_like(df.index, True, dtype=bool)
    for col, filered_text in text_filters.items():
        if filered_text.strip():
            mask = mask & df[col].str.lower().str.contains(filered_text.lower())

    # mask = category_text_filter(df, mask, columns_to_plot)

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
