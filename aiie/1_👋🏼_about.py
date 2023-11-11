import pandas as pd

import streamlit as st
from data import get_clean_data
from utils import github_repo_url

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
    made by **Djalel Benbouzid**, reach out on [LinkedIn](http://linkedin.com/in/dbenbouzid/) for questions and comments.
    """,
    icon="👋",
)

st.success(
    """
    **related publication**: [A sector-based approach to AI ethics: Understanding ethical issues of AI-related incidents within their sectoral context](https://dl.acm.org/doi/10.1145/3600211.3604680)
    """,
    icon="📄",
)


with st.expander("", expanded=True):
    cols = st.columns(5)
    cols[0].metric(":red[Total incidents]", df.index.size)
    cols[2].metric("Countries", df[C.country].nunique())
    cols[4].metric("Sectors", df[C.sector].nunique())

    st.plotly_chart(
        df[C.occurred]
        .value_counts()
        .rename("Incidents")
        .rename_axis(index="Year")
        .sort_index()
        .plot.area(line_shape="spline")
        .update_layout(showlegend=False),
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
    "http://localhost:8501/search",
    use_container_width=True,
    type="primary",
)
cols[2].link_button(
    "Talk to the incidents!",
    "http://localhost:8501/",
    use_container_width=True,
    type="primary",
    disabled=True,
    help="Soon!",
)

with st.sidebar.expander("Changelog", expanded=False):
    st.info("**v0.1**: AIIE launched.")


st.sidebar.warning(
    f"""
    something broken?
    [Let me know by opening a GitHub issue!]({github_repo_url})
    """,
    icon="👾",
)
st.sidebar.info(
    """
    **license**: CC BY-NC-SA 4.0
    """,
    icon="📝",
)
