from enum import StrEnum

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

AIAAIC_SHEET_ID = "1Bn55B4xz21-_Rgdr8BBb2lt0n_4rzLGxFADMlVW0PYI"
AIAAIC_SHEET_NAME = "Repository"


# Conveniency column enum
# Provides auto completion
# Allows the easy update of column names in case it changes in the source
# (which is beyond my control)
# Admittedly, it could have been replaced by
# df.columns.str.strip().str.lower().str.split('(').str[0].str.replace(r'\W', '_', regex=True)
class C(StrEnum):
    title = "Headline/title"
    type = "Type"
    released = "Released"
    occurred = "Occurred"
    country = "Country(ies)"
    sector = "Sector(s)"
    operator = "Operator(s)"
    developer = "Deployer(s)"
    system_name = "System name(s)"
    technology = "Technology(ies)"
    purpose = "Purpose(s)"
    media_trigger = "Media trigger(s)"
    risks = "Issue(s)"
    transparency = "Transparency"
    external_harms_individual = "External harms Individual"
    external_harms_societal = "External harms Societal"
    external_harms_environmental = "External harms Environmental"
    internal_harms_strategic_reputational = "Internal harms Strategic/reputational"
    internal_harms_strategic_operational = "Internal harms Operational"
    internal_harms_strategic_financial = "Internal harms Financial"
    internal_harms_strategic_Legal = "Internal harms Legal/regulatory"
    # operational = "Operational"
    # financial = "Financial"
    # societal = "Societal"
    # environmental = "Environmental"
    # legal_regulatory = "Legal/regulatory"
    summary_links = "Description/links"


# @st.cache_data(show_spinner="Fetchez la data... 🐮")
def read_gsheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url, header=0, skiprows=[0, 2], skip_blank_lines=True).dropna(
        how="all"
    )


@st.cache_data(show_spinner="Fetching more information about the incident...")
def scrap_incident_description(link):
    soup = BeautifulSoup(requests.get(link).text, "html.parser")

    # This is dangeriously hard-coded.
    description = soup.find_all(
        class_="hJDwNd-AhqUyc-uQSCkd Ft7HRd-AhqUyc-uQSCkd jXK9ad D2fZ2 zu5uec OjCsFc dmUFtb wHaque g5GTcb"
    )[1].get_text()
    return description


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # remove the extra unused columns
    cols_to_drop = [name for name in df.columns if name.startswith("Unnamed:")]

    # and remove the trailing spaces from the column names
    df = (
        df.set_index(df.columns[0])
        .drop(columns=cols_to_drop)
        .rename(columns=lambda x: x.strip())
    )

    # quick check that the column names did not change in the source repo
    original_column_names = df.columns.to_list()
    new_column_names = list(map(str, C))

    if original_column_names != new_column_names:
        st.toast(
            "Some columns appear to have changed in the AIAAIC repository, hence some parts of this app might not work properly."
        )
    df[C.country] = df[C.country].str.replace(";", ",")
    df[C.transparency] = df[C.transparency].str.replace(";", ",")
    df[C.risks] = df[C.risks].str.replace(";", ",")
    df[C.technology] = df[C.technology].str.replace(";", ",")

    # convert the years to Int16 (i.e. int16 with a None option)
    int_columns = {C.released, C.occurred}
    for col in int_columns:
        df[col] = df[col].astype(str).str.split(";").str[0]
        df[col] = df[col].astype(str).str.split("-").str[0]

        # df[col] = df[col].astype("Int64")

    # handle the categorical columns
    # cat_columns = {C.type, C.country, C.sector, C.technology, C.risks, C.transparency}
    # for col in cat_columns:
    #     df[col] = df[col].astype("category")

    # convert to string (better than the `object` type)

    str_columns = set(df.columns.to_list()) - int_columns  # - cat_columns
    for col in str_columns:
        df[col] = df[col].astype("string").fillna("Unknown")

    return df

def get_clean_data():
    # df = read_gsheet(AIAAIC_SHEET_ID, AIAAIC_SHEET_NAME)
    df = pd.read_csv("repository.csv").dropna(how="all")
    df = clean_data(df)
    # remove hidden columns
    df = df.drop(
        columns=[
            C.external_harms_individual,
            C.external_harms_societal,
            C.external_harms_environmental,
            C.internal_harms_strategic_reputational,
            C.internal_harms_strategic_operational,
            C.internal_harms_strategic_financial,
            C.internal_harms_strategic_Legal,
        ]
    )
    df.to_csv("aiie/pages/processed_dataset.csv", index=False)  # Save to the correct directory

    st.session_state["data"] = df
    st.session_state["columns"] = C

    return df, C

get_clean_data()

def prepare_topic_analysis(df, description):
    pass
