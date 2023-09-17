from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
import streamlit as st


def dataframe_with_selections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a checkbox column at the beginning of the dataframe.
    It is a trick while Streamlit has no other way to capture the row selection event.
    From https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
    """
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(required=True),
            "Summary/links": st.column_config.LinkColumn(),
            "Country(s)": st.column_config.ListColumn(),
        },
        disabled=df.columns,

    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.
    Adapted from: https://blog.streamlit.io/make-dynamic-filters-in-streamlit-and-show-their-effects-on-the-original-dataset/

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.sidebar.expander("Filtering")

    with modification_container:
        to_filter_columns = st.multiselect("Filter data on", df.columns)
        for column in to_filter_columns:
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = st.multiselect(
                    f"Values for :red[{column}]",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = df[column].min()
                _max = df[column].max()
                # step = (_max - _min) / 100
                user_num_input = st.slider(
                    f"Values for :red[{column}]",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    # step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = st.date_input(
                    f"Values for :red[{column}]",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = st.text_input(
                    f"Text search in :red[{column}]",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.lower().str.contains(user_text_input.lower())]

    return df
