import concurrent.futures
from datetime import date

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from tornado_data import (
    get_dat_tornadoes,
    get_states,
    get_counties,
    assign_states,
    assign_counties,
    create_counts,
    build_detail_table,
    get_tornado_photos,
    fetch_attachment_bytes,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
PHOTO_GRID_COLUMNS = 4
PHOTO_FETCH_WORKERS = 6


def _looks_like_image(photo):
    content_type = (photo["content_type"] or "").lower()
    name = (photo["name"] or "").lower()
    return content_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS)

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

TABLE_KEY = "state_table"
MAP_KEY = "state_map"


def build_choropleth(result, states):
    fig = px.choropleth(
        result,
        locations="STUSPS",
        locationmode="USA-states",
        color="tornadoes",
        scope="usa",
        hover_name="NAME",
        hover_data={"STUSPS": False, "tornadoes": True},
        custom_data=["STUSPS", "NAME"],
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

    # Clicking a state to open its drilldown also triggers Plotly's
    # built-in selection dimming, fading every other state (and their
    # count labels) with no way to reset it from this app. The
    # subheader/detail section below is already the "you selected this"
    # feedback, so force full opacity in both states everywhere instead.
    fig.update_traces(
        selector=dict(type="choropleth"),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=1)),
    )

    return fig


def on_table_select():
    rows = st.session_state[TABLE_KEY]["selection"]["rows"]
    if rows:
        st.session_state["selected_state"] = (
            st.session_state["result"]["STUSPS"].iloc[rows[0]]
        )


def on_map_select():
    points = st.session_state[MAP_KEY]["selection"]["points"]
    if points:
        st.session_state["selected_state"] = points[0]["customdata"][0]


st.set_page_config(page_title="Tornado Counts by State", layout="wide")

# Streamlit's built-in fullscreen-expand button on images is easy to
# miss - enlarge it. The toolbar is a *sibling* of stImage inside their
# shared stElementContainer (not a descendant of stImage), so scoping
# requires :has() on the shared ancestor rather than a plain descendant
# selector - otherwise this matches nothing. Scoped this way (rather
# than a bare stElementToolbarButton selector) so chart/dataframe
# toolbars, which reuse the same component, aren't also enlarged.
st.markdown(
    """
    <style>
    div[data-testid="stElementContainer"]:has(div[data-testid="stImage"])
            [data-testid="stElementToolbarButton"] {
        width: 2.25rem;
        height: 2.25rem;
    }
    div[data-testid="stElementContainer"]:has(div[data-testid="stImage"])
            svg[data-testid="stElementToolbarButtonIcon"] {
        width: 1.5rem;
        height: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# DAT survey data is actively revised, so cache it briefly. State and
# county boundaries are large, static files, so cache them for the
# session. Photo lookups are cheap to repeat but pointless to re-fetch
# on every rerun, so they're cached too.
get_dat_tornadoes = st.cache_data(
    ttl=900, show_spinner="Querying NWS Damage Assessment Toolkit..."
)(get_dat_tornadoes)
get_states = st.cache_data(
    ttl=None, show_spinner="Downloading U.S. state boundaries..."
)(get_states)
get_counties = st.cache_data(
    ttl=None, show_spinner="Downloading U.S. county boundaries..."
)(get_counties)
get_tornado_photos = st.cache_data(ttl=None, show_spinner=False)(get_tornado_photos)
fetch_attachment_bytes = st.cache_data(ttl=None, show_spinner=False)(fetch_attachment_bytes)

st.title("Tornado Counts by State")
st.caption(
    "Surveyed NWS tornadoes per state for a selected date range, from the "
    "Damage Assessment Toolkit. A tornado track that crosses a state line "
    "is counted once for each state it touches. Click a state in the "
    "table or on the map to see its individual tornadoes."
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
        counties = get_counties()
        intersections = assign_states(tornadoes, states)
        county_touches = assign_counties(tornadoes, counties, states)
        result = create_counts(intersections, states)
    except requests.RequestException as exc:
        st.error(f"Failed to reach the NWS DAT service: {exc}")
        st.stop()

    st.session_state.update(
        tornadoes=tornadoes,
        states=states,
        intersections=intersections,
        county_touches=county_touches,
        result=result,
        selected_state=None,
    )

if "result" in st.session_state:
    tornadoes = st.session_state["tornadoes"]
    states = st.session_state["states"]
    intersections = st.session_state["intersections"]
    county_touches = st.session_state["county_touches"]
    result = st.session_state["result"]

    total = int(result["tornadoes"].sum())

    if total == 0:
        st.info("No surveyed tornadoes in this range.")
    else:
        st.metric("Total tornado-state occurrences", total)

    table_col, map_col = st.columns([1, 2])

    with table_col:
        display_result = result.rename(
            columns={"NAME": "State", "tornadoes": "Tornadoes"}
        )[["State", "Tornadoes"]]
        st.dataframe(
            display_result,
            hide_index=True,
            use_container_width=True,
            on_select=on_table_select,
            selection_mode="single-row",
            key=TABLE_KEY,
        )

    with map_col:
        fig = build_choropleth(result, states)
        st.plotly_chart(
            fig,
            use_container_width=True,
            on_select=on_map_select,
            selection_mode="points",
            key=MAP_KEY,
        )

    selected = st.session_state.get("selected_state")

    if selected:
        state_name = states.set_index("STUSPS").loc[selected, "NAME"]
        event_ids = intersections.loc[
            intersections["STUSPS"] == selected, "event_id"
        ]
        detail_source = tornadoes[tornadoes["event_id"].isin(event_ids)]

        st.subheader(f"Tornadoes in {state_name}")

        if detail_source.empty:
            st.caption("No surveyed tornadoes for this state in the selected range.")
        else:
            detail = build_detail_table(detail_source, county_touches)

            for _, row in detail.iterrows():
                with st.expander(f"{row['Date']} — {row['EF Rating']} — {row['Counties']}"):
                    st.table(
                        row.drop(["event_id", "objectid", "Narrative"]).rename("Value")
                    )

                    with st.expander("Narrative", expanded=False):
                        st.write(row["Narrative"])

                    if st.button(
                        "Load photos", key=f"photos_{row['objectid']}"
                    ):
                        photos = get_tornado_photos(
                            row["event_id"], row["objectid"]
                        )
                        if not photos:
                            st.caption("No photos available for this survey.")
                        else:
                            image_photos = [p for p in photos if _looks_like_image(p)]
                            other_photos = [p for p in photos if not _looks_like_image(p)]

                            # Pre-create every grid cell, then fetch all
                            # photos concurrently and fill each cell in as
                            # soon as its download finishes, rather than
                            # downloading one at a time and blocking the
                            # whole grid on the slowest-to-load photo.
                            placeholders = {}
                            for i in range(0, len(image_photos), PHOTO_GRID_COLUMNS):
                                row_photos = image_photos[i:i + PHOTO_GRID_COLUMNS]
                                for col, photo in zip(st.columns(PHOTO_GRID_COLUMNS), row_photos):
                                    with col:
                                        placeholder = st.empty()
                                        placeholder.text(f"Loading {photo['name']}...")
                                        placeholders[photo["url"]] = placeholder

                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=PHOTO_FETCH_WORKERS
                            ) as executor:
                                future_to_photo = {
                                    executor.submit(fetch_attachment_bytes, photo["url"]): photo
                                    for photo in image_photos
                                }
                                for future in concurrent.futures.as_completed(future_to_photo):
                                    photo = future_to_photo[future]
                                    placeholder = placeholders[photo["url"]]
                                    data = future.result()
                                    if data:
                                        placeholder.image(
                                            data,
                                            caption=photo["name"],
                                            use_container_width=True,
                                        )
                                    else:
                                        placeholder.warning(f"Could not load: {photo['name']}")

                            for photo in other_photos:
                                st.markdown(f"[{photo['name']}]({photo['url']})")
