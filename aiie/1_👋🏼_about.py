import pandas as pd

import streamlit as st
from data import get_clean_data

pd.options.plotting.backend = "plotly"
st.set_page_config(
    page_title="AIIA - About",
    layout="centered",
    page_icon="img/logo.png",
    initial_sidebar_state="expanded",
)

# Get the data into df
# and the column mapper into C
df, C = get_clean_data()

# Header & logo
st.image("img/logo.png")
st.write(
    """
    # AI Incidents Explorer

    - Welcome to the AIIE! 👋 (pronounced "aïe!", which in french, means "ouch!").
    - This tool helps you learn more about the **actual risks of AI systems**,
    or at least, about the **past AI incidents** that found their way into the headlines.
    - It is built on top of the [AIAAIC](https://www.aiaaic.org/aiaaic-repository) repository by [Charlie Pownall](https://charliepownall.com) ✨
    """
)

st.success(
    """
    made by **Djalel Benbouzid**, reach out on [LinkedIn](http://linkedin.com/in/dbenbouzid/).
    """,
    icon="👋",
)

st.success(
    """
    **related publication**: [A sector-based approach to AI ethics: Understanding ethical issues of AI-related incidents within their sectoral context](https://dl.acm.org/doi/10.1145/3600211.3604680)
    """,
    icon="📄",
)

st.info(
    """
    **license**: CC BY-NC-SA 4.0
    """,
    icon="📝",
)

# st.write( """
#             <img width="50" height="50" src="img/cc-logo.f0ab4ebe.svg">
#             <img width="50" height="50" src="img/cc-by.21b728bb.svg">
#             <img width="50" height="50" src="img/cc-nc.218f18fc.svg">
#             <img width="50" height="50" src="img/cc-sa.d1572b71.svg">
#             """, unsafe_allow_html=True)

st.info(
    """
    something broken?
    [Let me know by opening a GitHub issue!](https://github.com/)
    """,
    icon="👾",
)

with st.expander("Summary", expanded=True):
    cols = st.columns([1, 1, 1, 2])
    cols[0].metric("Total incidents", df.index.size)
    cols[1].metric("Countries", df[C.country].nunique())
    cols[2].metric("Sectors", df[C.sector].nunique())
    cols[3].metric("Years", f"{df[C.occurred].min()} – {df[C.occurred].max()}")

    # with st.expander("Timeline - Number of reported incidents per year", expanded=True):
    st.plotly_chart(
        df[C.occurred]
        .value_counts()
        .rename("Incidents")
        .rename_axis(index="Year")
        .plot.bar(),
        use_container_width=True,
    )

# st.divider()

cols = st.columns(3)
cols[0].link_button(
    "See the incidents",
    "http://localhost:8501/plots",
    use_container_width=True,
    type="primary",
)
cols[1].link_button(
    "Read the incidents",
    "http://localhost:8501/plots",
    use_container_width=True,
    type="primary",
)
cols[2].link_button(
    "Talk to the incidents!",
    "http://localhost:8501/plots",
    use_container_width=True,
    type="primary",
    disabled=True,
    help="Soon!",
)

with st.expander("Changelog", expanded=False):
    st.info("**v0.1**: AIIE launched.")
