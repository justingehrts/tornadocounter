import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from io import BytesIO

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

YEAR = 2026

DAT_URL = (
    "https://services.dat.noaa.gov/arcgis/rest/services/"
    "nws_damageassessmenttoolkit/DamageViewer/FeatureServer/1/query"
)

STATES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/"
    "shp/cb_2025_us_state_20m.zip"
)

OUTPUT_FILE = "tornado_counts.csv"


# ------------------------------------------------------------
# Download tornado data from NWS Damage Assessment Toolkit
# ------------------------------------------------------------

def get_dat_tornadoes(year):
    print(f"Downloading {year} tornado data from NWS DAT...")

    where = (
        f"stormdate >= DATE '{year}-01-01' "
        f"AND stormdate < DATE '{year + 1}-01-01' "
        f"AND efscale IN ('EFU','EF0','EF1','EF2','EF3','EF3+','EF4','EF5')"
    )

    params = {
        "where": where,
        "outFields": (
            "event_id,stormdate,efscale,startlat,startlon,"
            "endlat,endlon,length,width"
        ),
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

        if not props.get("event_id") or not geom:
            continue

        features.append({
            **props,
            "geometry": shape(geom)
        })

    if not features:
        raise RuntimeError(
            "No tornadoes were returned from the DAT service."
        )

    tornadoes = gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs="EPSG:4326"
    )

    # DAT should normally have one damage line per event, but
    # make sure we don't accidentally count duplicate event IDs.
    tornadoes = tornadoes.drop_duplicates(
        subset="event_id"
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

    # Remove non-state territories and other areas that we don't
    # want in the 50-state count.
    states = states[
        states["STUSPS"].isin([
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
            "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
            "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
            "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
        ])
    ].copy()

    return states


# ------------------------------------------------------------
# Determine which states each tornado intersects
# ------------------------------------------------------------

def assign_states(tornadoes, states):
    print("Assigning tornadoes to states...")

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
# Create the final 50-state table
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
# Main program
# ------------------------------------------------------------

def main():

    tornadoes = get_dat_tornadoes(YEAR)

    states = get_states()

    intersections = assign_states(
        tornadoes,
        states
    )

    result = create_counts(
        intersections,
        states
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print(f"{YEAR} TORNADO COUNTS BY STATE")
    print("=" * 60)

    print(
        result.to_string(index=False)
    )

    print()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
