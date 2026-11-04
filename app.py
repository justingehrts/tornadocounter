import concurrent.futures
import io
from datetime import date, timedelta

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageOps

from tornado_data import (
    STATE_ABBRS,
    SYNTHETIC_EVENT_ID_PREFIX,
    DATServiceError,
    get_dat_tornadoes,
    get_states,
    get_counties,
    assign_states,
    assign_counties,
    create_counts,
    build_tornado_table,
    format_tornado_detail,
    get_tornado_photos,
    fetch_attachment_bytes,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
PHOTO_GRID_COLUMNS = 4
PHOTO_FETCH_WORKERS = 6

EF_OPTIONS = ["EF0", "EF1", "EF2", "EF3", "EF4", "EF5", "EFU"]

# Light -> dark by intensity; EFU (unrated) in neutral gray.
EF_TRACK_COLORS = {
    "EFU": "#7f7f7f",
    "EF0": "#feb24c",
    "EF1": "#fd8d3c",
    "EF2": "#f03b20",
    "EF3": "#bd0026",
    "EF4": "#800026",
    "EF5": "#49006a",
}

RANGE_YTD = "Year to date"
RANGE_7 = "Last 7 days"
RANGE_30 = "Last 30 days"
RANGE_CUSTOM = "Custom"

VIEW_COUNTS = "Counts"
VIEW_TRACKS = "Tracks"

TORNADO_TABLE_COLUMNS = [
    "Date", "Time", "EF Rating", "Counties", "Path Length (mi)",
    "Path Width (yd)", "Deaths", "Injuries", "Max Wind (mph)",
    "Property Damage ($K)", "NWS Office",
]

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


def _looks_like_image(photo):
    content_type = (photo["content_type"] or "").lower()
    name = (photo["name"] or "").lower()
    return content_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS)


def _ef_bucket(rating):
    return "EF3" if rating == "EF3+" else rating


def _keep_click_on_selected_state(fig):
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
    _keep_click_on_selected_state(fig)

    return fig


def build_tracks_map(result, tornadoes):
    # Neutral states underneath keep the map clickable for drilldown.
    fig = go.Figure(go.Choropleth(
        locations=result["STUSPS"],
        z=[0] * len(result),
        locationmode="USA-states",
        colorscale=[[0, "#eceff1"], [1, "#eceff1"]],
        showscale=False,
        customdata=result[["STUSPS", "NAME"]].to_numpy(),
        hovertemplate="%{customdata[1]}<extra></extra>",
        marker_line_color="white",
        marker_line_width=0.8,
    ))

    tracks_by_rating = {rating: ([], [], []) for rating in EF_TRACK_COLORS}
    for event_id, rating, local_date, geom in zip(
        tornadoes["event_id"], tornadoes["efscale"],
        tornadoes["local_date"], tornadoes.geometry,
    ):
        rating = _ef_bucket(rating)
        if rating not in tracks_by_rating or geom is None:
            continue
        name = (
            "Unnamed tornado"
            if str(event_id).startswith(SYNTHETIC_EVENT_ID_PREFIX)
            else event_id
        )
        label = f"{name}<br>{local_date} · {rating}"
        lats, lons, texts = tracks_by_rating[rating]
        for part in getattr(geom, "geoms", [geom]):
            xs, ys = part.xy
            lons += list(xs) + [None]
            lats += list(ys) + [None]
            texts += [label] * len(xs) + [None]

    # Weakest first, so the strongest tracks draw on top.
    for rating in ["EFU", "EF0", "EF1", "EF2", "EF3", "EF4", "EF5"]:
        lats, lons, texts = tracks_by_rating[rating]
        if not lats:
            continue
        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            text=texts,
            mode="lines",
            line=dict(width=2, color=EF_TRACK_COLORS[rating]),
            name=rating,
            hoverinfo="text",
        ))

    fig.update_layout(
        geo=dict(scope="usa"),
        legend=dict(title="EF rating"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    _keep_click_on_selected_state(fig)

    return fig


def on_table_select():
    rows = st.session_state[TABLE_KEY]["selection"]["rows"]
    if rows:
        st.session_state["selected_state"] = st.session_state["result_order"][rows[0]]


def on_map_select():
    # Only the state layer carries customdata; clicks on track lines or
    # count labels are ignored.
    for point in st.session_state[MAP_KEY]["selection"]["points"]:
        customdata = point.get("customdata")
        if customdata:
            st.session_state["selected_state"] = customdata[0]
            return


def _fetch_upright_photo(url):
    """Download a photo and apply its EXIF orientation.

    Phone photos are often stored sideways with an EXIF tag saying how to
    rotate them, which st.image ignores. Undecodable data is returned as is.
    """
    data = fetch_attachment_bytes(url)
    if not data:
        return data
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.getexif().get(0x0112, 1) == 1:
                return data
            upright = ImageOps.exif_transpose(img).convert("RGB")
            out = io.BytesIO()
            upright.save(out, format="JPEG", quality=90)
            return out.getvalue()
    except Exception:
        return data


def render_photos(event_id, objectid):
    photos = get_tornado_photos(event_id, objectid)
    if not photos:
        st.caption("No photos available for this survey.")
        return

    image_photos = [p for p in photos if _looks_like_image(p)]
    other_photos = [p for p in photos if not _looks_like_image(p)]

    # Pre-create every grid cell, then fetch all photos concurrently and
    # fill each cell in as soon as its download finishes, rather than
    # downloading one at a time and blocking the whole grid on the
    # slowest-to-load photo.
    placeholders = {}
    for i in range(0, len(image_photos), PHOTO_GRID_COLUMNS):
        row_photos = image_photos[i:i + PHOTO_GRID_COLUMNS]
        for col, photo in zip(st.columns(PHOTO_GRID_COLUMNS), row_photos):
            with col:
                placeholder = st.empty()
                placeholder.text(f"Loading {photo['name']}...")
                placeholders[photo["url"]] = placeholder

    with concurrent.futures.ThreadPoolExecutor(max_workers=PHOTO_FETCH_WORKERS) as executor:
        future_to_photo = {
            executor.submit(_fetch_upright_photo, photo["url"]): photo
            for photo in image_photos
        }
        for future in concurrent.futures.as_completed(future_to_photo):
            photo = future_to_photo[future]
            placeholder = placeholders[photo["url"]]
            data = future.result()
            if data:
                placeholder.image(data, caption=photo["name"], width="stretch")
            else:
                placeholder.warning(f"Could not load: {photo['name']}")

    for photo in other_photos:
        st.markdown(f"[{photo['name']}]({photo['url']})")


@st.cache_data(ttl=900, show_spinner="Loading tornado data from NOAA...")
def load_view(start_date, end_date):
    # DAT survey data is actively revised, so this is only cached briefly.
    # Census boundaries are cached on disk by tornado_data itself.
    tornadoes = get_dat_tornadoes(start_date, end_date)
    states = get_states()
    counties = get_counties()
    intersections = assign_states(tornadoes, states)
    county_touches = assign_counties(tornadoes, counties, states)
    return tornadoes, states, intersections, county_touches


get_tornado_photos = st.cache_data(ttl=None, show_spinner=False)(get_tornado_photos)
fetch_attachment_bytes = st.cache_data(ttl=None, show_spinner=False)(fetch_attachment_bytes)


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

st.title("Tornado Counts by State")
st.caption(
    "Surveyed NWS tornadoes per state for a selected date range, from the "
    "Damage Assessment Toolkit. Dates are each tornado's local date. A "
    "tornado track that crosses a state line is counted once for each "
    "state it touches. Click a state in the table or on the map to see "
    "its individual tornadoes."
)

today = date.today()
last_year = str(today.year - 1)

# Every control below is bound to the URL, so the address bar always
# reproduces the current view and can be bookmarked or shared.
range_choice = st.segmented_control(
    "Date range",
    [RANGE_YTD, RANGE_7, RANGE_30, last_year, RANGE_CUSTOM],
    default=RANGE_YTD,
    required=True,
    key="range",
    bind="query-params",
)

if range_choice == RANGE_CUSTOM:
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date", value=date(today.year, 1, 1),
            key="start", bind="query-params",
        )
    with col2:
        end_date = st.date_input(
            "End date", value=today, key="end", bind="query-params",
        )
    if start_date > end_date:
        st.error("Start date must not be after end date.")
        st.stop()
elif range_choice == RANGE_7:
    start_date, end_date = today - timedelta(days=6), today
elif range_choice == RANGE_30:
    start_date, end_date = today - timedelta(days=29), today
elif range_choice == last_year:
    start_date, end_date = date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
else:
    start_date, end_date = date(today.year, 1, 1), today

selected_efs = st.pills(
    "EF rating",
    EF_OPTIONS,
    selection_mode="multi",
    default=EF_OPTIONS,
    key="ef",
    bind="query-params",
)

try:
    tornadoes, states, intersections, county_touches = load_view(start_date, end_date)
except (requests.RequestException, DATServiceError) as exc:
    st.error(
        "Couldn't load data from NOAA or the Census Bureau. "
        f"Please try again in a moment.\n\nDetails: {exc}"
    )
    st.stop()

if not selected_efs:
    st.info("Select at least one EF rating to see tornadoes.")
    st.stop()

ef_mask = tornadoes["efscale"].map(_ef_bucket).isin(selected_efs)
tornadoes = tornadoes[ef_mask]
kept_ids = set(tornadoes["event_id"])
intersections = intersections[intersections["event_id"].isin(kept_ids)]
county_touches = county_touches[county_touches["event_id"].isin(kept_ids)]
result = create_counts(intersections, states)

# The table/map selection callbacks run before the next script run, so
# they need the row order as it was displayed.
st.session_state["result_order"] = result["STUSPS"].tolist()

if "selected_state" not in st.session_state:
    url_state = st.query_params.get("state")
    st.session_state["selected_state"] = url_state if url_state in STATE_ABBRS else None

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
        on_select=on_table_select,
        selection_mode="single-row",
        key=TABLE_KEY,
    )
    st.download_button(
        "Download state counts (CSV)",
        display_result.to_csv(index=False),
        file_name=f"tornado_counts_{start_date}_{end_date}.csv",
        mime="text/csv",
        on_click="ignore",
    )

with map_col:
    view = st.segmented_control(
        "Map view",
        [VIEW_COUNTS, VIEW_TRACKS],
        default=VIEW_COUNTS,
        required=True,
        key="view",
        bind="query-params",
    )
    if view == VIEW_TRACKS:
        fig = build_tracks_map(result, tornadoes)
    else:
        fig = build_choropleth(result, states)
    st.plotly_chart(
        fig,
        on_select=on_map_select,
        selection_mode="points",
        key=MAP_KEY,
    )

selected = st.session_state.get("selected_state")

if selected:
    st.query_params["state"] = selected
elif "state" in st.query_params:
    del st.query_params["state"]

if selected:
    state_name = states.set_index("STUSPS").loc[selected, "NAME"]
    event_ids = intersections.loc[intersections["STUSPS"] == selected, "event_id"]
    detail_source = tornadoes[tornadoes["event_id"].isin(event_ids)]

    st.subheader(f"Tornadoes in {state_name}")

    if detail_source.empty:
        st.caption("No surveyed tornadoes for this state in the selected range.")
    else:
        tornado_table = build_tornado_table(detail_source, county_touches)

        st.caption("Click a column header to sort. Select a row for details and photos.")
        selection = st.dataframe(
            tornado_table[TORNADO_TABLE_COLUMNS],
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"tornado_table_{selected}",
            column_config={
                "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Path Length (mi)": st.column_config.NumberColumn(format="%.2f"),
                "Path Width (yd)": st.column_config.NumberColumn(format="%d"),
                "Deaths": st.column_config.NumberColumn(format="%d"),
                "Injuries": st.column_config.NumberColumn(format="%d"),
                "Max Wind (mph)": st.column_config.NumberColumn(format="%d"),
                "Property Damage ($K)": st.column_config.NumberColumn(format="$%.0fK"),
            },
        )

        export_columns = [c for c in tornado_table.columns if not c.startswith("_")]
        st.download_button(
            f"Download {state_name} tornado list (CSV)",
            tornado_table[export_columns].to_csv(index=False),
            file_name=f"tornadoes_{selected}_{start_date}_{end_date}.csv",
            mime="text/csv",
            on_click="ignore",
        )

        rows = selection.selection.rows
        if rows:
            row = tornado_table.iloc[rows[0]]
            st.markdown(f"#### {row['Date']} — {row['EF Rating']} — {row['Counties']}")

            detail = format_tornado_detail(row)
            st.table({"Value": detail})

            with st.expander("Narrative", expanded=False):
                st.write(row["Narrative"] or "No narrative available.")

            if st.button("Load photos", key=f"photos_{row['_objectid']}"):
                render_photos(row["_event_id"], row["_objectid"])
