from datetime import date

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from tornado_data import (
    get_dat_tornadoes,
    get_states,
    assign_states,
    create_counts,
)

# States too small/crowded to hold a label at their true location get
# their count pulled out to a stacked column over the Atlantic, with a
# thin leader line back to the state. Order/positions are hand-tuned
# to read top-to-bottom roughly north-to-south along the coast.
SMALL_STATE_LABEL_OFFSETS = {
    "VT": (46.6, -68.5),
    "NH": (45.0, -67.0),
    "MA": (43.4, -65.5),
    "RI": (41.8, -64.3),
    "CT": (40.2, -64.8),
    "NJ": (38.6, -65.3),
    "DE": (37.0, -65.8),
    "MD": (35.4, -66.3),
    "DC": (33.8, -66.8),
}


def build_choropleth(result, states):
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

    # representative_point() (rather than centroid) guarantees a point
    # that actually falls inside each state's polygon, which matters for
    # irregular/multi-part shapes (e.g. Michigan, Florida, Hawaii).
    anchors = states.set_index("STUSPS").geometry.representative_point()

    max_count = result["tornadoes"].max()
    light_text_threshold = max_count * 0.55 if max_count > 0 else 0

    main_lats, main_lons, main_text, main_colors = [], [], [], []
    callout_lats, callout_lons, callout_text = [], [], []
    leader_lats, leader_lons = [], []

    for _, row in result.iterrows():
        stusps, count = row["STUSPS"], row["tornadoes"]
        anchor = anchors[stusps]

        if stusps in SMALL_STATE_LABEL_OFFSETS:
            label_lat, label_lon = SMALL_STATE_LABEL_OFFSETS[stusps]
            callout_lats.append(label_lat)
            callout_lons.append(label_lon)
            callout_text.append(f"{stusps}: {count}")
            leader_lats += [anchor.y, label_lat, None]
            leader_lons += [anchor.x, label_lon, None]
        else:
            main_lats.append(anchor.y)
            main_lons.append(anchor.x)
            main_text.append(str(count))
            main_colors.append(
                "white" if count > light_text_threshold else "#1a1a1a"
            )

    fig.add_trace(go.Scattergeo(
        lat=leader_lats,
        lon=leader_lons,
        mode="lines",
        line=dict(width=0.75, color="rgba(70,70,70,0.7)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scattergeo(
        lat=main_lats,
        lon=main_lons,
        text=main_text,
        mode="text",
        textfont=dict(size=10, color=main_colors),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scattergeo(
        lat=callout_lats,
        lon=callout_lons,
        text=callout_text,
        mode="text",
        textposition="middle right",
        textfont=dict(size=10, color="#1a1a1a"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        coloraxis_colorbar_title="Tornadoes",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig

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
        fig = build_choropleth(result, states)
        st.plotly_chart(fig, use_container_width=True)
