import time
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape
from timezonefinder import TimezoneFinder

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DAT_BASE_URL = (
    "https://services.dat.noaa.gov/arcgis/rest/services/"
    "nws_damageassessmenttoolkit/DamageViewer/FeatureServer"
)
DAT_URL = f"{DAT_BASE_URL}/1/query"

# The layer's maxRecordCount; larger result sets must be paged.
DAT_PAGE_SIZE = 2000

STATES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/"
    "shp/cb_2025_us_state_20m.zip"
)

COUNTIES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/"
    "shp/cb_2025_us_county_20m.zip"
)

# The URLs above pin a Census vintage, so a cached copy never goes stale.
CACHE_DIR = Path.home() / ".cache" / "tornadocounter"

RETRY_DELAY_SECONDS = 2

EF_SCALES = ("EFU", "EF0", "EF1", "EF2", "EF3", "EF3+", "EF4", "EF5")

DAT_FIELDS = [
    "event_id", "objectid", "stormdate", "starttime", "endtime",
    "efscale", "startlat", "startlon", "endlat", "endlon",
    "length", "width", "wfo", "fatalities", "injuries", "maxwind",
    "cropdamage", "propdamage", "comments",
]
LOCAL_TIME_FIELDS = ["tz", "local_start", "local_end", "local_date"]

# Many DAT records have a blank event_id; get_dat_tornadoes substitutes
# this prefix plus the objectid so every row has a unique id.
SYNTHETIC_EVENT_ID_PREFIX = "DAT-"

STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

_tz_finder = TimezoneFinder()


class DATServiceError(RuntimeError):
    """The DAT service answered, but with an error instead of data."""


# ------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------

def _get(url, params=None, timeout=60):
    """GET with one retry on connection problems, timeouts and HTTP 5xx."""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        if status is not None and status < 500:
            raise

    time.sleep(RETRY_DELAY_SECONDS)
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def _dat_json(url, params, timeout=60):
    # ArcGIS reports query errors as HTTP 200 with an "error" body.
    data = _get(url, params=params, timeout=timeout).json()
    if "error" in data:
        error = data["error"]
        raise DATServiceError(
            f"DAT service error {error.get('code')}: {error.get('message')}"
        )
    return data


def _cached_download(url):
    """Local path to the file at url, downloading it on first use."""
    path = CACHE_DIR / url.rsplit("/", 1)[-1]
    if path.exists():
        return path

    response = _get(url, timeout=120)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(path)
    return path


# ------------------------------------------------------------
# Download tornado data from NWS Damage Assessment Toolkit
# ------------------------------------------------------------

def _query_all_features(params):
    features = []
    offset = 0

    while True:
        page = _dat_json(
            DAT_URL,
            {
                **params,
                "orderByFields": "objectid",
                "resultOffset": offset,
                "resultRecordCount": DAT_PAGE_SIZE,
            },
            timeout=120,
        )
        if "features" not in page:
            raise DATServiceError(f"DAT did not return features: {page}")

        batch = page["features"]
        features.extend(batch)

        exceeded = (
            page.get("exceededTransferLimit")
            or page.get("properties", {}).get("exceededTransferLimit")
        )
        if not batch or (not exceeded and len(batch) < DAT_PAGE_SIZE):
            return features

        offset += len(batch)


def _epoch_ms_to_utc(values):
    return pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit="ms", utc=True)


def _add_local_times(tornadoes):
    """
    DAT times are UTC, but NWS reports tornadoes by local calendar date:
    an evening storm in Ohio is the 26th locally and the 27th in UTC.
    Convert each track's times to the time zone of its starting point.
    """
    zones = []
    for lat, lon, geom in zip(
        tornadoes["startlat"], tornadoes["startlon"], tornadoes.geometry
    ):
        if pd.isna(lat) or pd.isna(lon):
            point = geom.centroid
            lat, lon = point.y, point.x
        zones.append(_tz_finder.timezone_at(lat=float(lat), lng=float(lon)) or "UTC")

    start_utc = _epoch_ms_to_utc(tornadoes["starttime"].fillna(tornadoes["stormdate"]))
    end_utc = _epoch_ms_to_utc(tornadoes["endtime"])

    tornadoes["tz"] = zones
    tornadoes["local_start"] = [
        ts.tz_convert(zone) if pd.notna(ts) else pd.NaT
        for ts, zone in zip(start_utc, zones)
    ]
    tornadoes["local_end"] = [
        ts.tz_convert(zone) if pd.notna(ts) else pd.NaT
        for ts, zone in zip(end_utc, zones)
    ]
    tornadoes["local_date"] = [
        ts.date() if pd.notna(ts) else None for ts in tornadoes["local_start"]
    ]
    return tornadoes


def _empty_tornadoes():
    return gpd.GeoDataFrame(
        columns=DAT_FIELDS + LOCAL_TIME_FIELDS + ["geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def get_dat_tornadoes(start_date, end_date):
    """Tornadoes whose local start date falls within [start_date, end_date]."""
    print(f"Downloading tornado data from NWS DAT for {start_date} to {end_date}...")

    # stormdate is UTC. Pad the server-side window so tornadoes whose local
    # date is in range but whose UTC date isn't are still fetched; the
    # exact local-date filter is applied below.
    where = (
        f"stormdate >= DATE '{(start_date - timedelta(days=1)).isoformat()}' "
        f"AND stormdate < DATE '{(end_date + timedelta(days=2)).isoformat()}' "
        f"AND efscale IN {EF_SCALES}"
    )

    raw_features = _query_all_features({
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    })

    features = []
    for feature in raw_features:
        props = feature.get("properties", {})
        geom = feature.get("geometry")

        if not geom:
            continue

        # About a quarter of real records have a blank event_id, and a
        # few share a placeholder like "1", so event_id can't be trusted
        # as a key. Fall back to the always-unique objectid.
        event_id = (
            props.get("event_id")
            or f"{SYNTHETIC_EVENT_ID_PREFIX}{props.get('objectid')}"
        )

        features.append({**props, "event_id": event_id, "geometry": shape(geom)})

    if not features:
        print("No tornadoes were returned from the DAT service for this range.")
        return _empty_tornadoes()

    tornadoes = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")

    # Guarantee every expected column exists, even if a field is absent
    # from every returned feature.
    tornadoes = tornadoes.reindex(columns=DAT_FIELDS + ["geometry"], fill_value=pd.NA)
    tornadoes = tornadoes.drop_duplicates(subset="objectid")

    tornadoes = _add_local_times(tornadoes)
    in_range = [
        local_date is not None and start_date <= local_date <= end_date
        for local_date in tornadoes["local_date"]
    ]
    tornadoes = tornadoes[in_range].reset_index(drop=True)

    if tornadoes.empty:
        print("No tornadoes were returned from the DAT service for this range.")
        return _empty_tornadoes()

    print(f"Found {len(tornadoes)} tornado events.")

    return tornadoes


# ------------------------------------------------------------
# Download U.S. state boundaries
# ------------------------------------------------------------

def get_states():
    print("Loading U.S. state boundaries...")

    states = gpd.read_file(_cached_download(STATES_URL))

    # Census ships this in NAD83 (EPSG:4269); DAT's geometry is WGS84.
    # sjoin across mismatched CRSs compares raw coordinates as if they
    # were the same system, so reproject at the source.
    states = states.to_crs(epsg=4326)

    # Remove territories and other areas that we don't want
    # in the state-by-state count.
    states = states[states["STUSPS"].isin(STATE_ABBRS)].copy()

    # STATEFP (state FIPS code) is kept so counties can be matched back
    # to a state abbreviation without a separate lookup table.
    return states[["STUSPS", "NAME", "STATEFP", "geometry"]].copy()


# ------------------------------------------------------------
# Determine which states each tornado intersects
# ------------------------------------------------------------

def assign_states(tornadoes, states):
    print("Assigning tornadoes to states...")

    if tornadoes.empty:
        return pd.DataFrame(columns=["event_id", "STUSPS", "NAME"])

    # Use an intersection rather than the tornado's starting point.
    intersections = gpd.sjoin(
        tornadoes[["event_id", "geometry"]],
        states[["STUSPS", "NAME", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    # A tornado can cross a state boundary, so an event can
    # legitimately appear in more than one state.
    intersections = intersections[
        ["event_id", "STUSPS", "NAME"]
    ].drop_duplicates()

    return intersections


# ------------------------------------------------------------
# Create the final state table
# ------------------------------------------------------------

def create_counts(intersections, states):
    print("Calculating state totals...")

    counts = (
        intersections
        .groupby(["STUSPS", "NAME"])
        .size()
        .reset_index(name="tornadoes")
    )

    # Make sure every state appears, even if it has zero tornadoes.
    result = states[["STUSPS", "NAME"]].copy()

    result = result.merge(
        counts,
        on=["STUSPS", "NAME"],
        how="left"
    )

    result["tornadoes"] = (
        result["tornadoes"]
        .fillna(0)
        .astype(int)
    )

    result = result.sort_values(
        "tornadoes",
        ascending=False
    ).reset_index(drop=True)

    return result


# ------------------------------------------------------------
# Download U.S. county boundaries
# ------------------------------------------------------------

def get_counties():
    print("Loading U.S. county boundaries...")

    counties = gpd.read_file(_cached_download(COUNTIES_URL))

    # Same NAD83 -> WGS84 reprojection as get_states().
    counties = counties.to_crs(epsg=4326)

    return counties[["STATEFP", "NAME", "NAMELSAD", "geometry"]].copy()


# ------------------------------------------------------------
# Determine which counties each tornado intersects
# ------------------------------------------------------------

def assign_counties(tornadoes, counties, states):
    print("Assigning tornadoes to counties...")

    if tornadoes.empty:
        return pd.DataFrame(columns=["event_id", "STUSPS", "NAME", "NAMELSAD"])

    intersections = gpd.sjoin(
        tornadoes[["event_id", "geometry"]],
        counties[["STATEFP", "NAME", "NAMELSAD", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    # Attach the state abbreviation via the state FIPS code, so
    # same-named counties in different states aren't ambiguous
    # (e.g. there are 30+ "Washington County"s).
    intersections = intersections.merge(
        states[["STATEFP", "STUSPS"]],
        on="STATEFP",
        how="left"
    )

    # A tornado can cross a county boundary, so an event can
    # legitimately appear in more than one county.
    intersections = intersections[
        ["event_id", "STUSPS", "NAME", "NAMELSAD"]
    ].drop_duplicates()

    return intersections


def counties_by_event(county_touches):
    """A "County, ST; County, ST" display string per event_id."""
    if county_touches.empty:
        return pd.Series(dtype=str, name="counties")

    return (
        county_touches
        .assign(label=lambda d: d["NAMELSAD"] + ", " + d["STUSPS"])
        .groupby("event_id")["label"]
        .apply(lambda labels: "; ".join(sorted(labels)))
    )


# ------------------------------------------------------------
# Tornado survey photos (ArcGIS attachments)
# ------------------------------------------------------------

def list_attachments(objectid, layer=1):
    """Attachment metadata for one DAT feature, or [] if there are none."""
    url = f"{DAT_BASE_URL}/{layer}/{objectid}/attachments"

    try:
        infos = _dat_json(url, {"f": "json"}, timeout=30).get("attachmentInfos", [])
    except (requests.RequestException, ValueError, DATServiceError):
        return []

    return [
        {
            "url": f"{url}/{info['id']}",
            "name": info.get("name"),
            "content_type": info.get("contentType"),
        }
        for info in infos
    ]


def get_tornado_photos(event_id, line_objectid):
    """
    Photo attachments for a tornado. Tries the track-line feature first
    (the layer this app already queries); some DAT surveys only attach
    photos to the individual damage-indicator points instead, so this
    falls back to that event's points if the line has none.
    """
    photos = list_attachments(line_objectid, layer=1)
    if photos or str(event_id).startswith(SYNTHETIC_EVENT_ID_PREFIX):
        # A synthetic id was never in DAT, so no point can match it.
        return photos

    escaped_event_id = str(event_id).replace("'", "''")

    try:
        data = _dat_json(
            f"{DAT_BASE_URL}/0/query",
            {
                "where": f"event_id = '{escaped_event_id}'",
                "outFields": "objectid",
                "returnGeometry": "false",
                "f": "json",
            },
            timeout=30,
        )
        point_ids = [f["attributes"]["objectid"] for f in data.get("features", [])]
    except (requests.RequestException, ValueError, KeyError, DATServiceError):
        return []

    combined = []
    for point_id in point_ids[:25]:
        combined.extend(list_attachments(point_id, layer=0))

    return combined


def fetch_attachment_bytes(url):
    """
    Download an attachment's raw bytes, or None on failure. Downloading
    server-side (rather than handing the URL straight to st.image, which
    fetches it client-side in the browser) uses the same requests session
    that's already proven able to reach this server for queries.
    """
    try:
        return _get(url, timeout=30).content
    except requests.RequestException:
        return None


# ------------------------------------------------------------
# Per-tornado table and detail formatting
# ------------------------------------------------------------

def _format_local_time(ts):
    if pd.isna(ts):
        return "—"
    return ts.strftime("%I:%M %p %Z").lstrip("0")


def _format_local_and_utc(ts):
    if pd.isna(ts):
        return "—"
    utc = ts.tz_convert("UTC")
    return f"{ts.date().isoformat()} {_format_local_time(ts)} ({utc.strftime('%H:%M')} UTC)"


def _format_number(value):
    if pd.isna(value):
        return "—"
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _format_thousands_of_dollars(value):
    # DAT stores damage in thousands of dollars (its own viewer labels
    # these fields "($k)"), so 800 means $800K.
    if pd.isna(value):
        return "—"
    return f"${value:,.0f}K" if value else "$0"


def build_tornado_table(tornadoes_subset, county_touches):
    """
    One row per tornado with typed columns, so the on-screen table sorts
    numerically/chronologically and the CSV export is analysis-ready.
    Columns starting with "_" are internal (photo lookup, detail card).
    """
    by_event = counties_by_event(county_touches)
    subset = tornadoes_subset.reset_index(drop=True)

    def numeric(column):
        # DAT records "not reported" as -99 (seen in damage, max wind and
        # width); no real value in these fields is ever negative.
        values = pd.to_numeric(subset[column], errors="coerce")
        return values.mask(values < 0)

    table = pd.DataFrame({
        "Date": subset["local_date"],
        "Time": [_format_local_time(ts) for ts in subset["local_start"]],
        "EF Rating": subset["efscale"].fillna("—"),
        "Counties": subset["event_id"].map(by_event).astype(object).fillna("—"),
        "Path Length (mi)": numeric("length").round(2),
        "Path Width (yd)": numeric("width"),
        "Deaths": numeric("fatalities"),
        "Injuries": numeric("injuries"),
        "Max Wind (mph)": numeric("maxwind"),
        "Property Damage ($K)": numeric("propdamage"),
        "Crop Damage ($K)": numeric("cropdamage"),
        "NWS Office": subset["wfo"].fillna("—"),
        "DAT Event Name": [
            "" if str(e).startswith(SYNTHETIC_EVENT_ID_PREFIX) else e
            for e in subset["event_id"]
        ],
        "Narrative": subset["comments"].fillna(""),
        "_event_id": subset["event_id"],
        "_objectid": subset["objectid"],
        "_local_start": subset["local_start"],
        "_local_end": subset["local_end"],
        "_start_utc": [
            ts.tz_convert("UTC") if pd.notna(ts) else pd.NaT
            for ts in subset["local_start"]
        ],
    })

    return table.sort_values(["Date", "_start_utc"]).reset_index(drop=True)


def format_tornado_detail(row):
    """Label -> display string for one row of build_tornado_table()."""
    return {
        "Date": row["Date"].isoformat() if pd.notna(row["Date"]) else "—",
        "Start": _format_local_and_utc(row["_local_start"]),
        "End": _format_local_and_utc(row["_local_end"]),
        "EF Rating": row["EF Rating"],
        "Counties": row["Counties"],
        "Path Length (mi)": _format_number(row["Path Length (mi)"]),
        "Path Width (yd)": _format_number(row["Path Width (yd)"]),
        "Deaths": _format_number(row["Deaths"]),
        "Injuries": _format_number(row["Injuries"]),
        "Max Wind (mph)": _format_number(row["Max Wind (mph)"]),
        "Property Damage": _format_thousands_of_dollars(row["Property Damage ($K)"]),
        "Crop Damage": _format_thousands_of_dollars(row["Crop Damage ($K)"]),
        "NWS Office": row["NWS Office"],
        "DAT Event Name": row["DAT Event Name"] or "—",
    }
