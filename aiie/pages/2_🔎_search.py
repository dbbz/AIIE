import pandas as pd
import streamlit as st
from data import get_clean_data
from utils import filter_dataframe, dataframe_with_filters


st.set_page_config(
    page_title="AIIA - Search",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)
pd.options.plotting.backend = "plotly"

col_1, col_2 = st.columns([5, 1])
col_1.title("🧭 AI Incidents Explorer")
st.divider()

df, C = get_clean_data()

st.sidebar.link_button(
    "Column descriptions",
    "https://www.aiaaic.org/aiaaic-repository/classifications-and-definitions#h.fyaxuf7wldm7",
    use_container_width=True,
    type="primary",
)

# Display the filtering widgets
df = dataframe_with_filters(
    df,
    on_columns=[C.type, C.country, C.sector, C.technology, C.risks, C.transparency],
    use_sidebar=True,
)
# df

# st.sidebar.download_button(
#     "Download filtered CSV",
#     df.to_csv().encode("utf-8"),
#     "aiaaic_filtered.csv",
#     "text/csv",
# )

# .style.where(
#         lambda val: any(term in str(val) for term in search if term),
#         "background-color: pink",
#     ),

st.data_editor(
    df,
    use_container_width=True,
    height=800,
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

# # Todo: filtering is not practical when too many categories and one just wants to select a few
# # Allow the section of rows (it's hack for now)
# df_selected = dataframe_with_selections(df, height=800)

# for link in df_selected[C.summary_links]:
#     st.write(link)
#     if link is not None:
#         description = scrap_incident_description(link)
#         st.write(description)
