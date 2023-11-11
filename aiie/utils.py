from collections import namedtuple
import numpy as np
import pandas as pd
import streamlit as st

# TODO: put the repo url here
github_repo_url = ""


def named_tabs(*tab_names):
    """
    A simple pattern for "named tabs",
    this way, one can add tabs on the fly in any order.
    """
    TabsNames = namedtuple("_", tab_names)
    tabs = st.tabs(TabsNames._fields)
    return TabsNames(*tabs)


# TODO: replace this filter with an event-based filtering + state management
# TODO: include nans too
def category_text_filter(
    df, mask, column_names, expander_label="Column filters", use_sidebar: bool = False
) -> np.ndarray:
    category_filters = {col: [] for col in column_names}

    expander_cls = st.sidebar.expander if use_sidebar else st.expander
    with expander_cls(expander_label, expanded=True):
        for col in column_names:
            counts = (
                df.loc[mask, col]
                .value_counts()
                .reset_index()
                .set_axis([col, "counts"], axis=1)
            )
            counts["labels"] = counts[col] + " (" + counts["counts"].astype(str) + ")"

            category_filters[col] = st.multiselect(
                col,
                counts[col].sort_values().unique(),
                format_func=lambda x: counts.set_index(col).loc[x, "labels"],
                key="cat_" + col,
            )
            if category_filters[col]:
                mask = mask & df[col].isin(category_filters[col])

    # for col, selected_values in category_filters.items():
    #     if selected_values:
    #         mask = mask & df[col].isin(selected_values)
    return mask


def dataframe_with_filters(
    df: pd.DataFrame,
    on_columns: list,
    use_sidebar: bool = False,
    with_col_filters: bool = True,
) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.

    Args:
        df (pd.DataFrame): Original dataframe
        on_columns (list of strings): The list of columns to filter on
        use_sidebar (bool): Whether to display the filtering widgets on the sidebar or the main page
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    mask = np.full_like(df.index, True, dtype=bool)

    with st.sidebar.expander("Global filter", expanded=True):
        search = st.text_input(
            "Enter keywords for search",
            help="Case-insensitive, comma-separated keywords. Prefix with ~ to exclude.",
        )

    search = [s.strip().lower() for s in search.split(",")]
    is_excluded = [s.startswith("~") for s in search]

    assert len(search) == len(is_excluded)

    search = [s.lstrip("~") for s in search]
    if search:
        for elem, exclude in zip(search, is_excluded):
            if elem and exclude:
                mask &= ~df.apply(
                    lambda row: row.astype(str).str.lower().str.contains(elem).any(),
                    axis=1,
                )
            else:
                mask &= df.apply(
                    lambda row: row.astype(str).str.lower().str.contains(elem).any(),
                    axis=1,
                )

    if with_col_filters:
        mask = category_text_filter(df, mask, on_columns, use_sidebar=use_sidebar)

    return df[mask]


@st.cache_data
def retain_most_frequent_values(df: pd.DataFrame, col: str, N: int) -> pd.DataFrame:
    top_N_values = (
        df[col].value_counts(sort=True, ascending=True).iloc[-N:].index.to_list()
    )
    return df[df[col].isin(top_N_values)]


def plot_counts(df, column, top_N):
    df_filtered = retain_most_frequent_values(df, column, top_N)
    df_counts = (
        df_filtered[column]
        .value_counts(sort=True, ascending=True)
        .to_frame(name="count")
    )
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
