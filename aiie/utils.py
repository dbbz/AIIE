import base64
from collections import namedtuple
from pathlib import Path
import re
import html2text
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from markdownify import markdownify
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

TTL = 30 * 60 * 24

# TODO: put the repo url here
github_repo_url = "https://github.com/dbbz/AIIE/issues"
deploy_url = "https://aiiexp.streamlit.app/"


@st.cache_data(ttl=TTL, show_spinner="Fetching more information about the incident...")
def scrap_incident_description(link):
    soup = BeautifulSoup(requests.get(link).text, "html.parser")

    # This is dangeriously hard-coded.
    description = soup.find_all(
        # class_="hJDwNd-AhqUyc-uQSCkd Ft7HRd-AhqUyc-uQSCkd purZT-AhqUyc-II5mzb ZcASvf-AhqUyc-II5mzb pSzOP-AhqUyc-qWD73c Ktthjf-AhqUyc-qWD73c JNdkSc SQVYQc"
        class_="hJDwNd-AhqUyc-uQSCkd Ft7HRd-AhqUyc-uQSCkd jXK9ad D2fZ2 zu5uec OjCsFc dmUFtb wHaque g5GTcb"
    )

    header_pattern = r"^(#+)\s+(.*)"
    description = markdownify("\n".join((str(i) for i in description[1:-1])))
    description = re.sub(header_pattern, r"#### \2", description)

    description = description.replace(
        "](/aiaaic-repository",
        "](https://www.aiaaic.org/aiaaic-repository",
    ).replace("### ", "##### ")
    return description


@st.cache_data(ttl=TTL, show_spinner="Fetching the list of links on the incident...")
def get_list_of_links(page_url):
    soup = BeautifulSoup(requests.get(page_url).text, "html.parser")
    section = soup.find(string=re.compile(", commentar"))

    if not section:
        section = soup.find(string=re.compile("act check 🚩"))
    if section:
        li_list = section.find_next("ul").find_all("li")
        # results = markdownify("\n".join(str(i) for i in li_list))
        results = [html2text.html2text(str(i)) for i in li_list]

        pattern = r"\[([^][]*)\]\(([^()]*)\)"
        urls = []
        for link in results:
            match = re.search(pattern, link)
            if match:
                urls.append(match.group(2))

        # links = [get_deepest_text(li) for li in li_list]
        return urls
    else:
        return []


# this function comes from Streamlit-Extra
def add_logo(logo_url: str, height: int = 120):
    """Add a logo (from logo_url) on the top of the navigation page of a multipage app.
    Taken from [the Streamlit forum](https://discuss.streamlit.io/t/put-logo-and-title-above-on-top-of-page-navigation-in-sidebar-of-multipage-app/28213/6)
    The url should be a local path to the image.

    Args:
        logo_url (str): URL/local path of the logo
    """

    logo = f"url(data:image/png;base64,{base64.b64encode(Path(logo_url).read_bytes()).decode()})"

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: {logo};
                background-repeat: no-repeat;
                padding-top: {height}px;
                background-position: 20px 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
def category_text_filter(df, mask, column_names) -> np.ndarray:
    category_filters = {col: [] for col in column_names}
    df = df[mask]
    # with st.expander("Filter by category", expanded=True):
    with st.container():
        st.caption("Filter by category")
        for col in column_names:
            if is_numeric_dtype(df[col]):
                min_value = df.loc[mask, col].min()
                max_value = df.loc[mask, col].max()

                min_max = st.slider(
                    col,
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                )
                mask = mask & df[col].between(*min_max)
            else:
                # df[col] = df[col].fillna("None")  # TMP
                counts = (
                    df.loc[mask, col]
                    .dropna()
                    .value_counts(dropna=False)
                    .reset_index()
                    .set_axis([col, "counts"], axis=1)
                )
                # counts.fillna("Unknown", inplace=True)
                # st.sidebar.write(counts)
                counts["labels"] = (
                    counts[col] + " (" + counts["counts"].astype(str) + ")"
                )
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


# make_dataframe_filters()
def dataframe_with_filters(
    df: pd.DataFrame, on_columns: list, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.

    Args:
        df (pd.DataFrame): Original dataframe
        on_columns (list of strings): The list of columns to filter on
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    mask = np.full_like(df.index, True, dtype=bool) if mask is None else mask

    search = st.text_input(
        "Search for specific terms",
        placeholder="Enter comma-separated keywords...",
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

    return mask


@st.cache_data
def retain_most_frequent_values(df: pd.DataFrame, col: str, N: int) -> pd.DataFrame:
    top_N_values = (
        df[col]
        .value_counts(sort=True, ascending=True, dropna=True)
        .iloc[-N:]
        .index.to_list()
    )
    return df[df[col].isin(top_N_values)]


def plot_counts(df, column, top_N):
    # This retains only the top N values of the column based on frequency
    df_filtered = retain_most_frequent_values(df, column, top_N)

    # Generate the counts
    df_counts = df_filtered[column].value_counts().reset_index()
    df_counts.columns = [column, "count"]

    # Sort the DataFrame by count to plot
    df_counts = df_counts.sort_values(by="count", ascending=True)

    # Creating a Plotly bar chart
    fig = px.bar(
        df_counts,
        x="count",
        y=column,
        orientation="h",
        title=f"Top {top_N} {column} by count",
    )
    st.plotly_chart(fig, use_container_width=True)


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
