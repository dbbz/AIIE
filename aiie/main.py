from collections import namedtuple
import pandas as pd
import streamlit as st

from utils import filter_dataframe

pd.options.plotting.backend = "plotly"

AIAAIC_SHEET_ID = '1Bn55B4xz21-_Rgdr8BBb2lt0n_4rzLGxFADMlVW0PYI'
AIAAIC_SHEET_NAME = 'Repository'


st.set_page_config(page_title="AIIA", layout="wide")
st.write("# AI Incidents Explorer")

@st.cache_data
def read_gsheet(sheet_id: str, sheet_url: str) -> pd.DataFrame:
    url = f'https://docs.google.com/spreadsheets/d/{AIAAIC_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={AIAAIC_SHEET_NAME}'
    return pd.read_csv(url).dropna(how="all")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # last_col_idx = data.columns.get_loc("Summary/links")
    cols_to_drop = [name for name in df.columns if name.startswith("Unnamed:")]
    return df.drop(columns=cols_to_drop)

with st.status("Fetchez la data...", expanded=False) as status:
    data = read_gsheet(AIAAIC_SHEET_ID, AIAAIC_SHEET_NAME)
    data = clean_data(data)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    status.update(label="Data downloaded!", state="complete", expanded=False)


st.dataframe(filter_dataframe(data), use_container_width=True, hide_index=True)

# url = "https://www.aiaaic.org/aiaaic-repository/ai-and-algorithmic-incidents-and-controversies/uk-visa-applications-filtering-racism"

# A simple pattern for "named tabs",
# this way, one can add tabs on the fly in any order.
TabsNames = namedtuple('_', ["page", "sectors", "released"])
tabs = st.tabs(TabsNames._fields)
tabs = TabsNames(*tabs)


with tabs.sectors:
    st.plotly_chart(data["Sector(s)"].hist(), use_container_width=True)

# st.plotly_chart(data.Released.hist(), use_container_width=True)
# st.plotly_chart(data.Occurred.hist(), use_container_width=True)
# st.plotly_chart(data.Type.hist(), use_container_width=True)

# with tabs.page:
    # data.iloc[0]
