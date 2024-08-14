from enum import StrEnum

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

AIAAIC_SHEET_ID = "1Bn55B4xz21-_Rgdr8BBb2lt0n_4rzLGxFADMlVW0PYI"
AIAAIC_SHEET_NAME = "Repository"

TTL = 30 * 60 * 24


# Conveniency column enum
# Provides auto completion
# Allows the easy update of column names in case it changes in the source
# (which is beyond my control)
# Admittedly, it could have been replaced by
# df.columns.str.strip().str.lower().str.split('(').str[0].str.replace(r'\W', '_', regex=True)
class C(StrEnum):
    title = "Headline"
    type = "Type"
    released = "Released"
    occurred = "Occurred"
    country = "Country(ies)"
    sector = "Sector(s)"
    operator = "Deployer(s)"
    developer = "Developer(s)"
    system_name = "System name(s)"
    technology = "Technology(ies)"
    purpose = "Purpose(s)"
    media_trigger = "Media trigger(s)"
    risks = "Issue(s)"
    transparency = "Transparency"

    # external_harms_individual = "External harms Individual"
    # external_harms_societal = "External harms Societal"
    # external_harms_environmental = "External harms Environmental"
    # internal_harms_strategic_reputational = "Internal harms Strategic/reputational"
    # internal_harms_strategic_operational = "Internal harms Operational"
    # internal_harms_strategic_financial = "Internal harms Financial"
    # internal_harms_strategic_Legal = "Internal harms Legal/regulatory"

    # operational = "Operational"
    # financial = "Financial"
    # societal = "Societal"
    # environmental = "Environmental"
    # legal_regulatory = "Legal/regulatory"

    summary_links = "Description/links"


# Load the actual AIAAIC repository (list of incidents)
# It used to be downloaded from the online repo
# but due to frequent changes in the sheet format
# I ended up using an offline (potentially not up to date) version
def get_repository_data():
    try:
        download_public_sheet_as_csv(
            "https://docs.google.com/spreadsheets/d/1Bn55B4xz21-_Rgdr8BBb2lt0n_4rzLGxFADMlVW0PYI/export?format=csv&gid=888071280"
        )
    except requests.exceptions.RequestException as e:
        st.toast(
            "The online repository could not be downloaded. Using a potentially old version."
        )
        # st.error(f"An error occurred: {e}")

    df = (
        pd.read_csv("downloaded_sheet.csv", skip_blank_lines=True, skiprows=[0, 2])
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    # df = df.set_index(df.columns[0]).rename(columns=lambda x: x.strip())

    return df


@st.cache_data(ttl=TTL, show_spinner="Fetchez la data... 🐮")
def download_public_sheet_as_csv(csv_url, filename="downloaded_sheet.csv"):
    """Downloads a public Google Sheet as a CSV file.

    Args:
        csv_url (str): The CSV download URL of the Google Sheet.
        filename (str, optional): The filename for the downloaded CSV file. Defaults to "downloaded_sheet.csv".
    """
    response = requests.get(csv_url)
    response.raise_for_status()  # Check for HTTP errors

    with open(filename, "wb") as f:
        f.write(response.content)


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

    df[C.country] = df[C.country].str.replace(";", ",")
    df[C.transparency] = df[C.transparency].str.replace(";", ",")
    df[C.risks] = df[C.risks].str.replace(";", ",")
    df[C.technology] = df[C.technology].str.replace(";", ",")

    # convert the years to Int16 (i.e. int16 with a None option)
    int_columns = {C.released, C.occurred}
    for col in int_columns:
        df[col] = df[col].astype(str).str.split(";").str[0]
        df[col] = df[col].astype(str).str.split("-").str[0]

        df[col] = pd.to_numeric(df[col], errors="coerce")
        # df[col] = df[col].astype("Int64")

    # handle the categorical columns
    # cat_columns = {C.type, C.country, C.sector, C.technology, C.risks, C.transparency}
    # for col in cat_columns:
    #     df[col] = df[col].astype("category")

    # convert to string (better than the `object` type)

    # str_columns = set(df.columns.to_list()) - int_columns  # - cat_columns
    # for col in str_columns:
    #     df[col] = df[col].astype("string").fillna("Unknown")

    return df


def get_clean_data(file_path="repository.csv"):
    # df = read_gsheet(AIAAIC_SHEET_ID, AIAAIC_SHEET_NAME)
    df = get_repository_data().dropna(how="all")

    columns_to_keep = list(map(str, C))
    df = clean_data(df)[columns_to_keep]

    # remove hidden columns
    # df = df.drop(
    #     columns=[
    #         C.external_harms_individual,
    #         C.external_harms_societal,
    #         C.external_harms_environmental,
    #         C.internal_harms_strategic_reputational,
    #         C.internal_harms_strategic_operational,
    #         C.internal_harms_strategic_financial,
    #         C.internal_harms_strategic_Legal,
    #     ]
    # )
    # df.to_csv(
    #     "aiie/pages/processed_dataset.csv", index=False
    # )  # Save to the correct directory

    st.session_state["data"] = df
    st.session_state["columns"] = C

    return df, C


def prepare_topic_analysis(df, description):
    pass
