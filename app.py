from datetime import date

import plotly.express as px
import requests
import streamlit as st

from tornado_data import (
    get_dat_tornadoes,
    get_states,
    assign_states,
    create_counts,
)

st.set_page_config(page_title="Tornado Counts by State", layout="wide")

# DAT survey data is actively revised, so cache it briefly. State
# boundaries are a large, static file, so cache them for the session.
get_dat_tornadoes = st.cache_data(
    ttl=900, show_spinner="Querying NWS Damage Assessment Toolkit..."
)(get_dat_tornadoes)
get_states = st.cache_data(
    ttl=None, show_spinner="Downloading U.S. state boundaries..."
)(get_states)

st.title("Tornado Counts by State")
st.caption(
    "Surveyed NWS tornadoes per state for a selected date range, from the "
    "Damage Assessment Toolkit. A tornado track that crosses a state line "
    "is counted once for each state it touches."
)

today = date.today()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=date(today.year, 1, 1))
with col2:
    end_date = st.date_input("End date", value=today)

if start_date > end_date:
    st.error("Start date must not be after end date.")
    st.stop()

if st.button("Get counts", type="primary"):
    try:
        tornadoes = get_dat_tornadoes(start_date, end_date)
        states = get_states()
        intersections = assign_states(tornadoes, states)
        result = create_counts(intersections, states)
    except requests.RequestException as exc:
        st.error(f"Failed to reach the NWS DAT service: {exc}")
        st.stop()

    total = int(result["tornadoes"].sum())

    if total == 0:
        st.info("No surveyed tornadoes in this range.")
    else:
        st.metric("Total tornado-state occurrences", total)

    table_col, map_col = st.columns([1, 2])

    with table_col:
        st.dataframe(result, hide_index=True, use_container_width=True)

    with map_col:
        fig = px.choropleth(
            result,
            locations="STUSPS",
            locationmode="USA-states",
            color="tornadoes",
            scope="usa",
            hover_name="NAME",
            hover_data={"STUSPS": False, "tornadoes": True},
            labels={"tornadoes": "Tornadoes"},
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            coloraxis_colorbar_title="Tornadoes",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
