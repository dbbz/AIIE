import pandas as pd
import streamlit as st

from data import get_clean_data
from utils import github_repo_url

pd.options.plotting.backend = "plotly"


# df = st.session_state.data
# C = st.session_state.columns

df, C = get_clean_data()

st.write(
    """
    # AI Incidents Explorer

    - Welcome to the AIIE! 👋 (pronounced "aïe!", which in french, means "ouch!").
    - This tool helps you learn more about the **actual risks of AI systems**,
    or at least, about the **past AI incidents** that found their way into the headlines.
    - It is built on top of the [AIAAIC](https://www.aiaaic.org/aiaaic-repository) repository ✨
    """
)

st.divider()

with st.container(border=False):
    cols = st.columns(3, gap="large")
    cols[0].metric("Total incidents", df.index.size)
    cols[1].metric("Countries", df[C.country].nunique())
    cols[2].metric("Sectors", df[C.sector].nunique())

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

cols = st.columns(2)

with cols[0]:
    st.success(
        """
        **related publications**:
        - [A sector-based approach to AI ethics: Understanding ethical issues of AI-related incidents within their sectoral context](https://dl.acm.org/doi/10.1145/3600211.3604680)
        - [A Collaborative, Human-Centred Taxonomy of AI, Algorithmic, and Automation Harms](https://arxiv.org/abs/2407.01294)
        """,
        icon="📄",
    )

with cols[1]:
    st.success(
        """
        **Author**: Djalel Benbouzid ([questions and comments](http://linkedin.com/in/dbenbouzid/))

        **Contributors**:
        - Sofia Vei, Aristotle University of Thessaloniki, Greece
        """,
        icon="👋",
    )

# cols = st.columns(3)
# cols[0].link_button(
#     "See the incidents",
#     f"{deploy_url}plots",
#     use_container_width=True,
#     type="primary",
# )
# cols[1].link_button(
#     "Read the incidents",
#     f"{deploy_url}search",
#     use_container_width=True,
#     type="primary",
# )
# cols[2].link_button(
#     "Talk to the incidents!",
#     f"{deploy_url}",
#     use_container_width=True,
#     type="primary",
#     disabled=True,
#     help="Soon!",
# )

with st.sidebar:
    with st.expander("Changelog", expanded=False):
        st.info("**v0.5**: Interactive heatmap plots added (thanks to Sofia).")
        st.info("**v0.4**: Topic analysis added.")
        st.info("**v0.3**: Sankey plots added.")
        st.info("**v0.2**: Search and plotting combined.")
        st.info("**v0.1**: AIIE launched.")

    st.warning(
        f"""
        something broken?
        [Let me know by opening a GitHub issue!]({github_repo_url})
        """,
        icon="👾",
    )

    st.warning(
        """
        This is still experimental and bugs are likely to exist. Please use with caution and scrupulously verify your analyses.
        """,
        icon="⚠️",
    )

    st.info(
        """
        **license**: CC BY-NC-SA 4.0
        """,
        icon="📝",
    )
