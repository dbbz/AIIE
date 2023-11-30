import numpy as np
import pandas as pd
import streamlit as st
from data import get_clean_data
from utils import (
    dataframe_with_filters,
    github_repo_url,
    retain_most_frequent_values,
    _df_groupby,
    gen_sankey,
    plot_counts,
    add_logo,
)
import shelve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


st.set_page_config(
    page_title="AIIA - Search",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)
add_logo("img/logo.png", 90)
pd.options.plotting.backend = "plotly"

query_parameters = st.experimental_get_query_params()

col_1, col_2 = st.columns([5, 1])
col_1.title("🧭 AI Incidents Explorer")

df, C = get_clean_data()
table_height = 400

# columns_to_plot = list(map(str, C))  # get all the column names
available_columns_to_plot = [
    C.country,
    C.type,
    C.sector,
    C.developer,
    C.technology,
    C.risks,
    C.transparency,
    C.media_trigger,
    C.developer,
    C.operator,
    C.purpose,
]


with st.sidebar.expander("Plotting", expanded=True):
    default_value = query_parameters.get("Columns", C.country)
    columns_to_plot = st.multiselect(
        "Columns",
        available_columns_to_plot,
        default_value,
        label_visibility="collapsed",
    )
    # top_N = st.number_input("Number of top values to show", 0, 20, 10)
    top_N = st.select_slider(
        "Show most frequent...", [5, 10, 15, 20, 25, 30, 40, 50, "all"], 25
    )

    # if st.sidebar.button("Save"):
    #     st.experimental_set_query_params(plotted=columns_to_plot)

    cols = st.columns(2)
    enable_sankey = cols[0].toggle("Sankey plot", False)
    enable_topic_modeling = cols[1].toggle("Topic analysis", False)

    n_components = 9
    if enable_topic_modeling:
        n_components = st.slider("Number of topics", 1, 20, value=9, step=1)

    tabs_names = columns_to_plot
    if enable_sankey:
        tabs_names += ["Sankey plot"]
    if enable_topic_modeling:
        tabs_names += ["Topic analysis"]

# Display the filtering widgets
df = dataframe_with_filters(
    df,
    on_columns=[
        # C.occurred,
        # C.released,
        C.type,
        C.country,
        C.sector,
        C.technology,
        C.risks,
        C.transparency,
        C.media_trigger,
    ],
    use_sidebar=True,
)


st.data_editor(
    df,
    use_container_width=True,
    height=table_height,
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

if tabs_names:
    table_height = 600
    plots_tabs = st.tabs(tabs_names)
    for i, col in enumerate(tabs_names):
        if col == "Sankey plot":
            with plots_tabs[i]:
                sankey_vars = st.multiselect(
                    "Choose at least two columns to plot",
                    available_columns_to_plot,
                    default=available_columns_to_plot[:2],
                    max_selections=4,
                    help="💡 Use the text filters for better plots.",
                )

                if sankey_vars:
                    sankey_cols = st.columns(len(sankey_vars))
                text_filters = {}
                for i, col in enumerate(sankey_vars):
                    text_filters[col] = sankey_cols[i].text_input(
                        "Text filter on " + col,
                        key="text_" + col,
                        help="Case-insensitive text filtering.",
                    )

                if len(sankey_vars) == 1:
                    st.warning("Select a second column to plot.", icon="⚠️")

                mask = np.full_like(df.index, True, dtype=bool)
                for col, filered_text in text_filters.items():
                    if filered_text.strip():
                        mask = mask & df[col].str.lower().str.contains(
                            filered_text.lower()
                        )

                if len(sankey_vars) > 1:
                    df_mask = df[mask]
                    df_sankey = _df_groupby(df_mask, sankey_vars)
                    fig = gen_sankey(
                        df_sankey,
                        sankey_vars,
                        "counts",
                        None,
                        # " - ".join(sankey_vars),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        elif col == "Topic analysis":
            pass
        else:
            if top_N != "all":
                df_filtered = retain_most_frequent_values(df, col, int(top_N))
            else:
                df_filtered = df
            count_plot = (
                df_filtered[col]
                .value_counts(sort=True, ascending=True)
                .to_frame(name="count")
                .plot(kind="barh")
                .update_layout(showlegend=False)
            )
            plots_tabs[i].plotly_chart(count_plot, use_container_width=True)


if enable_topic_modeling:
    with shelve.open("description", "r") as db:
        df_description = pd.DataFrame.from_dict(
            db, orient="index", columns=["Description"]
        )

        df_description = df.join(df_description).dropna()
        df_description = df_description[C.title] + ", " + df_description.Description

    if not df_description.empty:
        incident_texts = df_description.to_list()
        n_top_words = 10

        tf_vectorizer = CountVectorizer(stop_words="english")
        tf = tf_vectorizer.fit_transform(incident_texts)
        feature_names = tf_vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=42,
        ).fit(tf)

        # topics = {}
        # for topic_idx, topic in enumerate(lda.components_):
        #     topics[f"Topic {topic_idx + 1}"] = [
        #         feature_names[i]
        #         for i in topic.argsort()[: -n_top_words - 1 : -1]
        #     ]

        # df_topics = pd.DataFrame(topics)
        # st.dataframe(df_topics, use_container_width=True)

        # topics_tabs = st.tabs([f"Topic {i + 1}" for i in range(n_components)])

        with plots_tabs[-1]:
            n_cols = 3
            for idx, topic in enumerate(lda.components_):
                if idx % n_cols == 0:
                    topics_cols = st.columns(n_cols)
                top_features_ind = topic.argsort()[-n_top_words:]
                top_features = feature_names[top_features_ind]
                weights = topic[top_features_ind]
                df_topic = pd.DataFrame(
                    {"word": top_features, "weight": weights}
                ).set_index("word")
                topics_cols[idx % n_cols].plotly_chart(
                    df_topic.plot.barh(title=f"Topic {idx + 1}").update_layout(
                        showlegend=False
                    ),
                    use_container_width=True,
                )
