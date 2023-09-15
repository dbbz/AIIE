import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import requests

from utils import filter_dataframe, dataframe_with_selections

pd.options.plotting.backend = "plotly"

AIAAIC_SHEET_ID = '1Bn55B4xz21-_Rgdr8BBb2lt0n_4rzLGxFADMlVW0PYI'
AIAAIC_SHEET_NAME = 'Repository'

st.set_page_config(page_title="AIIA - Search", layout="wide", page_icon="🔎", initial_sidebar_state="expanded")

col_1, col_2 = st.columns([5, 1])
col_1.title("🧭 AI Incidents Explorer")
st.divider()
@st.cache_data
def read_gsheet(sheet_id: str, sheet_url: str) -> pd.DataFrame:
    url = f'https://docs.google.com/spreadsheets/d/{AIAAIC_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={AIAAIC_SHEET_NAME}'
    return pd.read_csv(url).dropna(how="all")

@st.cache_data
def scrap_incident_description(link):
    soup = BeautifulSoup(requests.get(link).text, 'html.parser')

    # This is dangeriously hard-coded.
    description = soup.find_all(class_="hJDwNd-AhqUyc-uQSCkd Ft7HRd-AhqUyc-uQSCkd jXK9ad D2fZ2 zu5uec OjCsFc dmUFtb wHaque g5GTcb")[1].get_text()
    return description

# TODO: Move this into data.py
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # remove the trailing spaces from the column names
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # remove the extra unused columns
    cols_to_drop = [name for name in df.columns if name.startswith("Unnamed:")]

    columns = set(df.columns.to_list())
    int_columns = {"Released", "Occurred"}
    cat_columns = {"Type", "Country(s)", "Sector(s)", "Technology(ies)"}
    str_columns = columns - int_columns - cat_columns

    # convert the years to Int16 (i.e. int16 with a None option)
    for col in int_columns:
        df[col] = df[col].astype('Int16')

    # handle the categorical columns
    for col in cat_columns:
        df[col] = df[col].astype('category')

    # convert to string (better than the `object` type)
    for col in str_columns:
        df[col] = df[col].astype('string')

    df.set_index(df.columns[0], inplace=True)
    # df["Released"] = df["Released"].astype('int16')
    return df.drop(columns=cols_to_drop)

with st.spinner("Fetchez la data... 🐮") as status:
    df = read_gsheet(AIAAIC_SHEET_ID, AIAAIC_SHEET_NAME)
    df = clean_data(df)
    st.session_state['data'] = df
    # status.update(label="Data downloaded!", state="complete", expanded=False)

col_2.metric("Total", df.index.size)

df = filter_dataframe(df)
df_selected = dataframe_with_selections(df)
# st.dataframe(df, use_container_width=True, hide_index=True)


for link in df_selected["Summary/links"]:
    st.write(link)
    if link is not None:
        description = scrap_incident_description(link)
        st.write(description)
