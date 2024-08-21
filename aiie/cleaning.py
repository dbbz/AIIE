import shelve
from itertools import chain

import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup
from data import get_clean_data, scrap_incident_description
from utils import named_tabs, dataframe_with_filters


# Password protect to avoid annotation vandalism
def check_password():
    """Returns `True` if the user had the correct password."""

    import hmac

    if "debug_mode" in st.secrets and st.secrets["debug_mode"]:
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

df, C = get_clean_data(raw=True)

# tabs = named_tabs("Countries", "Other")

# with tabs.Countries:

col = st.selectbox("Select a column to investigate", df.columns)

counts = df[col].value_counts().to_frame().reset_index()
mask = dataframe_with_filters(
    counts, on_columns=counts.columns, text="Filter on specific categories"
)
counts = counts[mask]

st.dataframe(
    counts,
    use_container_width=True,
    hide_index=True,
    column_config={"count": st.column_config.Column(width=200)},
    height=700,
)

# with tabs.Other:
#     unique_values = df[C.country].unique()
#     unique_values = (
#         pd.Series(unique_values, dtype="string")
#         .str.replace(";", ",")
#         .str.strip()
#         .str.strip(",")
#         .str.split(",")
#         .dropna()
#     )
#     unique_values = pd.Series(sorted(list(chain.from_iterable(unique_values))))
#     unique_values = unique_values.value_counts().sort_index()

#     st.dataframe(unique_values, width=500, height=700)


def explore_clean(df):
    from cleaning import cleaning_functions, splitting_functions

    selected_col_to_plot = st.sidebar.selectbox("Column", list_of_cols_to_plot, index=2)

    df_clean = df.copy()
    df_clean[selected_col_to_plot] = (
        df[selected_col_to_plot]
        .str.lower()
        .str.strip()
        .to_frame(name=selected_col_to_plot)
    )

    unique_values = pd.Series(
        df_clean[selected_col_to_plot].sort_values().unique(), name=selected_col_to_plot
    )

    col_1, col_2 = st.columns([2, 3])
    col_1.subheader(f'How we are cleaning the "{selected_col_to_plot}" categories')
    unique_values_title = col_2.empty()

    filered_text = st.sidebar.text_input(f"Filter categories on {selected_col_to_plot}")

    cleaning_func = cleaning_functions.get(selected_col_to_plot, lambda x, with_echo: x)
    splitting_func = splitting_functions.get(selected_col_to_plot, lambda x: x)

    with col_1:
        # unique_values = unique_values.str.lower().str.strip()
        cleaning_button = st.sidebar.checkbox("Apply cleaning instructions", value=True)
        with st.expander("Cleaning instructions", expanded=False):
            if cleaning_button:
                unique_values = cleaning_func(unique_values, with_echo=True)
                df_clean[selected_col_to_plot] = cleaning_func(
                    df_clean[selected_col_to_plot], with_echo=False
                )

                unique_values = unique_values.str.strip("%").str.split("%").explode()
                df_clean[selected_col_to_plot] = (
                    df_clean[selected_col_to_plot].str.strip("%").str.split("%")
                )
                df_clean = df_clean.explode(selected_col_to_plot)

                unique_values = unique_values.str.strip()
                df_clean[selected_col_to_plot] = df_clean[
                    selected_col_to_plot
                ].str.strip()
                unique_values = pd.Series(unique_values.sort_values().unique())

        unique_values = unique_values.to_frame(name=selected_col_to_plot)
        split_columns = st.sidebar.checkbox("Apply column splitting", value=False)

        with st.expander("Column splitting instructions", expanded=False):
            if split_columns:
                unique_values = splitting_func(unique_values, with_echo=True)
                df_clean = df_clean.reset_index()
                df_clean[selected_col_to_plot] = splitting_func(
                    df_clean[[selected_col_to_plot]]
                )[selected_col_to_plot]

                # unique_values = pd.Series(
                #     (unique_values[selected_col_to_plot].sort_values().unique())
                # )
                # unique_values = unique_values.to_frame(name=selected_col_to_plot)

    unique_values_title.subheader(
        f'"{selected_col_to_plot}" currently has {unique_values[selected_col_to_plot].size} unique values'
    )

    if filered_text.strip():
        mask = (
            df_clean[selected_col_to_plot]
            .str.lower()
            .str.contains(filered_text.lower())
        )
        df_clean = df_clean[mask]

    fig, counts = plot_value_counts(df_clean, selected_col_to_plot)
    with col_1:
        with st.expander("Count plot", expanded=True):
            st.plotly_chart(fig, use_container_width=True)
    with col_2:
        with st.expander("Sort", expanded=False):
            sort_type = st.radio("Sort", ["alphabetically", "by frequency"])
        with st.expander("Count table", expanded=True):
            if sort_type == "alphabetically":
                st.dataframe(counts.sort_index(), height=1200)
            elif sort_type == "by frequency":
                st.dataframe(
                    counts.sort_values(by="counts", ascending=False), height=1200
                )

        # plot_with_table(fig, counts, selected_col_to_plot)

    if filered_text.strip():
        mask = (
            unique_values[selected_col_to_plot]
            .str.lower()
            .str.contains(filered_text.lower())
        )
        unique_values = unique_values[mask]

    if split_columns:
        with st.expander("Unique values after splitting the column", expanded=False):
            n_split_cols = unique_values.shape[1]
            cols = st.columns(n_split_cols + 2)
            cols[0].dataframe(unique_values, height=500)
            cols[1].write(
                "<h3 style='text-align: center;'>is split into</h3>",
                unsafe_allow_html=True,
            )
            for i, c in enumerate(unique_values.columns, 2):
                cols[i].dataframe(unique_values[c].sort_values().unique(), height=500)

    assert np.array_equal(
        unique_values[selected_col_to_plot].sort_values().unique(),
        df_clean[selected_col_to_plot].sort_values().unique(),
    )

    if st.sidebar.checkbox("Show samples", True):
        with st.expander(f"Samples for {selected_col_to_plot}", expanded=True):
            values_selected = st.multiselect(
                "values",
                df_clean[selected_col_to_plot].sort_values().unique(),
            )
            if values_selected:
                df_sample = df_clean[
                    df_clean[selected_col_to_plot].isin(values_selected)
                ]
                AgGrid(df_sample.sample(min(5, df_sample.shape[0])))
                st.write(f"Out of {df_sample.shape[0]} row(s).")


def plot_value_counts(data, col, height=None):
    if isinstance(data, pd.DataFrame):
        data = data[col]
    counts = (
        data.value_counts()
        .to_frame()
        .reset_index()
        .set_axis([col, "counts"], axis=1)
        .set_index(col)
    )
    size = counts.shape[0]
    height = height or (size * 25)
    fig = px.bar(counts.sort_values(by="counts"), x="counts", height=height)
    return fig, counts
