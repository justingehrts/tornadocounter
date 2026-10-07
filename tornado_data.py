import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from datetime import timedelta
from io import BytesIO

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DAT_BASE_URL = (
    "https://services.dat.noaa.gov/arcgis/rest/services/"
    "nws_damageassessmenttoolkit/DamageViewer/FeatureServer"
)
DAT_URL = f"{DAT_BASE_URL}/1/query"

STATES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/"
    "shp/cb_2025_us_state_20m.zip"
)

COUNTIES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/"
    "shp/cb_2025_us_county_20m.zip"
)

EF_SCALES = ("EFU", "EF0", "EF1", "EF2", "EF3", "EF3+", "EF4", "EF5")

# Fields confirmed working today.
DAT_CORE_FIELDS = [
    "event_id", "stormdate", "efscale", "startlat", "startlon",
    "endlat", "endlon", "length", "width",
]
# Fields expected on this layer based on the DAT service's published
# schema, but not verified against a live response from this sandbox
# (network egress to services.dat.noaa.gov is blocked here). If any of
# these come back empty across every tornado, check the real field
# names at DAT_BASE_URL + "/1?f=json" and update this list.
DAT_EXTENDED_FIELDS = [
    "objectid", "wfo", "fatalities", "injuries",
    "maxwind", "cropdamage", "propdamage",
    "starttime", "endtime", "comments",
]
DAT_FIELDS = DAT_CORE_FIELDS + DAT_EXTENDED_FIELDS

# Detail-table column labels and formatting, keyed by the DAT field name.
# "comments" (the survey narrative) is deliberately not listed here - it's
# long-form text, shown separately rather than as a table row.
DETAIL_COLUMN_LABELS = {
    "stormdate": "Date",
    "starttime": "Start Time (UTC)",
    "endtime": "End Time (UTC)",
    "efscale": "EF Rating",
    "length": "Path Length (mi)",
    "width": "Path Width (yd)",
    "fatalities": "Deaths",
    "injuries": "Injuries",
    "maxwind": "Max Wind (mph)",
    "propdamage": "Property Damage",
    "cropdamage": "Crop Damage",
    "wfo": "NWS Office",
}
CURRENCY_FIELDS = {"propdamage", "cropdamage"}
NUMERIC_FIELDS = {"length", "width", "fatalities", "injuries", "maxwind"}

STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


# ------------------------------------------------------------
# Download tornado data from NWS Damage Assessment Toolkit
# ------------------------------------------------------------

def get_dat_tornadoes(start_date, end_date):
    print(f"Downloading tornado data from NWS DAT for {start_date} to {end_date}...")

    where = (
        f"stormdate >= DATE '{start_date.isoformat()}' "
        f"AND stormdate < DATE '{(end_date + timedelta(days=1)).isoformat()}' "
        f"AND efscale IN {EF_SCALES}"
    )

    params = {
        "where": where,
        # "*" rather than an explicit list: some of the fields we want
        # (deaths, injuries, damage, etc.) are not verified against a
        # live response, and an ArcGIS FeatureServer can reject a query
        # outright if outFields names a field that doesn't exist.
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    response = requests.get(DAT_URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "features" not in data:
        raise RuntimeError(
            f"DAT did not return features:\n{data}"
        )

    features = []

    for feature in data["features"]:
        props = feature.get("properties", {})
        geom = feature.get("geometry")

        if not geom:
            continue

        # About a quarter of 2026's real records have a blank event_id
        # (and a few share a non-unique one, e.g. "1") - skipping or
        # collapsing those by event_id silently discarded hundreds of
        # genuinely distinct tornadoes. Fall back to the one field
        # that's always present and unique (objectid) so every row
        # downstream that keys off event_id - including this
        # function's own dedup below - treats each tornado as distinct.
        event_id = props.get("event_id") or f"DAT-{props.get('objectid')}"

        features.append({
            **props,
            "event_id": event_id,
            "geometry": shape(geom)
        })

    if not features:
        print("No tornadoes were returned from the DAT service for this range.")
        return gpd.GeoDataFrame(
            columns=DAT_FIELDS + ["geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )

    tornadoes = gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs="EPSG:4326"
    )

    # Guarantee every expected column exists, even if a field is absent
    # from every returned feature (a plain missing key doesn't become a
    # NaN column in a dict-built DataFrame - it just doesn't exist).
    tornadoes = tornadoes.reindex(
        columns=DAT_FIELDS + ["geometry"], fill_value=pd.NA
    )

    # Guard against the query somehow returning the same feature twice.
    # Dedup on objectid, not event_id: a real 2026 nationwide pull shows
    # roughly 1 in 4 tornadoes has a blank event_id (and a handful of
    # others share a non-unique one, e.g. "1"), so deduping on event_id
    # silently collapsed hundreds of genuinely distinct tornadoes down
    # to one. objectid is the service's actual unique row identifier -
    # always present, always unique.
    tornadoes = tornadoes.drop_duplicates(
        subset="objectid"
    )

    print(f"Found {len(tornadoes)} tornado events.")

    return tornadoes


# ------------------------------------------------------------
# Download U.S. state boundaries
# ------------------------------------------------------------

def get_states():
    print("Downloading U.S. state boundaries...")

    response = requests.get(STATES_URL, timeout=60)
    response.raise_for_status()

    states = gpd.read_file(
        BytesIO(response.content)
    )

    # Census ships this in NAD83 (EPSG:4269); DAT's geometry is WGS84
    # (EPSG:4326, requested via outSR in get_dat_tornadoes). sjoin
    # against mismatched CRSs silently compares raw coordinates as if
    # they were the same system, which is wrong even though the two
    # systems are close enough in the continental US that it rarely
    # flips a result - reproject at the source so every consumer of
    # this GeoDataFrame (assign_states, the map's representative_point
    # anchors, etc.) is consistently in WGS84.
    states = states.to_crs(epsg=4326)

    # Remove territories and other areas that we don't want
    # in the state-by-state count.
    states = states[
        states["STUSPS"].isin(STATE_ABBRS)
    ].copy()

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
    print("Downloading U.S. county boundaries...")

    response = requests.get(COUNTIES_URL, timeout=60)
    response.raise_for_status()

    counties = gpd.read_file(
        BytesIO(response.content)
    )

    # Same NAD83 -> WGS84 reprojection as get_states(), for the same
    # reason: DAT geometry is WGS84, and assign_counties sjoins against
    # this directly.
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
        response = requests.get(url, params={"f": "json"}, timeout=30)
        response.raise_for_status()
        infos = response.json().get("attachmentInfos", [])
    except (requests.RequestException, ValueError):
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
    (the layer this app already queries); some DAT surveys may only
    attach photos to the individual damage-indicator points instead, so
    this falls back to checking that event's points if the line has none.
    """
    photos = list_attachments(line_objectid, layer=1)
    if photos:
        return photos

    try:
        response = requests.get(
            f"{DAT_BASE_URL}/0/query",
            params={
                "where": f"event_id = '{event_id}'",
                "outFields": "objectid",
                "returnGeometry": "false",
                "f": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        point_ids = [
            f["attributes"]["objectid"]
            for f in response.json().get("features", [])
        ]
    except (requests.RequestException, ValueError, KeyError):
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
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


# ------------------------------------------------------------
# Build the per-tornado detail table for a selected state
# ------------------------------------------------------------

def _format_currency(value):
    if pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def _format_number(value):
    if pd.isna(value):
        return "—"
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _to_timestamp(value):
    """
    DAT date/time fields have come back as epoch-milliseconds integers
    rather than ISO strings, despite querying with f=geojson. Treating a
    millisecond value as nanoseconds (pandas' default for a bare number)
    collapses any real date to just after 1970-01-01, so the unit has to
    be set explicitly whenever the value is numeric.
    """
    if pd.isna(value):
        return None
    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit="ms")
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        return None


def _format_date(value):
    ts = _to_timestamp(value)
    if ts is not None:
        return ts.date().isoformat()
    return "—" if pd.isna(value) else str(value)


def _format_datetime_utc(value):
    ts = _to_timestamp(value)
    if ts is not None:
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return "—" if pd.isna(value) else str(value)


def build_detail_table(tornadoes_subset, county_touches):
    columns = list(DETAIL_COLUMN_LABELS)

    detail = tornadoes_subset[["event_id", "objectid", "comments"] + columns].copy()

    detail["counties"] = detail["event_id"].map(
        counties_by_event(county_touches)
    ).fillna("—")

    detail["Narrative"] = detail["comments"].apply(
        lambda value: str(value) if pd.notna(value) and str(value).strip()
        else "No narrative available."
    )
    detail = detail.drop(columns=["comments"])

    detail["stormdate"] = detail["stormdate"].apply(_format_date)

    for field in ("starttime", "endtime"):
        detail[field] = detail[field].apply(_format_datetime_utc)

    for field in CURRENCY_FIELDS:
        detail[field] = pd.to_numeric(detail[field], errors="coerce").apply(_format_currency)

    for field in NUMERIC_FIELDS:
        detail[field] = pd.to_numeric(detail[field], errors="coerce").apply(_format_number)

    detail["efscale"] = detail["efscale"].fillna("—")
    detail["wfo"] = detail["wfo"].fillna("—")

    detail = detail.rename(columns=DETAIL_COLUMN_LABELS)
    detail = detail.rename(columns={"counties": "Counties"})

    detail = detail.sort_values("Date").reset_index(drop=True)

    return detail
