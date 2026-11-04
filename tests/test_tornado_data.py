from datetime import date, datetime, timezone

import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import LineString, box

import tornado_data as td


class FakeResponse:
    def __init__(self, payload=None, status_code=200, content=b""):
        self._payload = payload
        self.status_code = status_code
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def epoch_ms(*args):
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


def feature(objectid, event_id="", start_ms=None, lat=40.65, lon=-83.6, efscale="EF1"):
    start_ms = start_ms if start_ms is not None else epoch_ms(2026, 3, 26, 18, 0)
    return {
        "type": "Feature",
        "properties": {
            "objectid": objectid,
            "event_id": event_id,
            "stormdate": start_ms,
            "starttime": start_ms,
            "endtime": start_ms + 60_000,
            "efscale": efscale,
            "startlat": lat,
            "startlon": lon,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat], [lon + 0.01, lat + 0.01]],
        },
    }


def page(features, exceeded=False):
    return {"type": "FeatureCollection", "features": features,
            "properties": {"exceededTransferLimit": True} if exceeded else {}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(td.time, "sleep", lambda seconds: None)


def serve_pages(monkeypatch, pages):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        offset = params.get("resultOffset", 0)
        return FakeResponse(pages[offset])

    monkeypatch.setattr(td.requests, "get", fake_get)
    return calls


# ------------------------------------------------------------
# Counting
# ------------------------------------------------------------

def test_track_crossing_a_state_line_counts_for_both_states():
    states = gpd.GeoDataFrame({
        "STUSPS": ["IN", "OH", "KY"],
        "NAME": ["Indiana", "Ohio", "Kentucky"],
        "geometry": [box(-88, 38, -85, 41), box(-85, 38, -82, 41), box(-88, 36, -82, 38)],
    }, crs="EPSG:4326")
    tornadoes = gpd.GeoDataFrame({
        "event_id": ["crosses IN->OH", "KY only"],
        "geometry": [LineString([(-86, 39.5), (-84, 39.5)]), LineString([(-86, 37), (-85, 37)])],
    }, crs="EPSG:4326")

    result = td.create_counts(td.assign_states(tornadoes, states), states)
    counts = dict(zip(result["STUSPS"], result["tornadoes"]))

    assert counts == {"IN": 1, "OH": 1, "KY": 1}


# ------------------------------------------------------------
# Fetching
# ------------------------------------------------------------

def test_blank_event_ids_stay_distinct(monkeypatch):
    serve_pages(monkeypatch, {0: page([feature(1, ""), feature(2, ""), feature(3, None)])})

    tornadoes = td.get_dat_tornadoes(date(2026, 3, 26), date(2026, 3, 26))

    assert len(tornadoes) == 3
    assert tornadoes["event_id"].is_unique


def test_paging_fetches_every_page(monkeypatch):
    monkeypatch.setattr(td, "DAT_PAGE_SIZE", 2)
    calls = serve_pages(monkeypatch, {
        0: page([feature(1), feature(2)], exceeded=True),
        2: page([feature(3), feature(4)], exceeded=True),
        4: page([feature(5)]),
    })

    tornadoes = td.get_dat_tornadoes(date(2026, 3, 26), date(2026, 3, 26))

    assert sorted(tornadoes["objectid"]) == [1, 2, 3, 4, 5]
    assert [c["resultOffset"] for c in calls] == [0, 2, 4]
    assert all(c["orderByFields"] == "objectid" for c in calls)


def test_error_body_raises_dat_service_error(monkeypatch):
    monkeypatch.setattr(td.requests, "get", lambda *a, **k: FakeResponse(
        {"error": {"code": 400, "message": "Invalid query"}}
    ))

    with pytest.raises(td.DATServiceError, match="Invalid query"):
        td.get_dat_tornadoes(date(2026, 1, 1), date(2026, 1, 31))


def test_retries_once_after_a_server_error(monkeypatch):
    responses = [FakeResponse(status_code=503), FakeResponse({"ok": True})]
    monkeypatch.setattr(td.requests, "get", lambda *a, **k: responses.pop(0))

    assert td._get("https://example.test").json() == {"ok": True}


def test_does_not_retry_a_client_error(monkeypatch):
    responses = [FakeResponse(status_code=404), FakeResponse({"ok": True})]
    monkeypatch.setattr(td.requests, "get", lambda *a, **k: responses.pop(0))

    with pytest.raises(requests.HTTPError):
        td._get("https://example.test")


# ------------------------------------------------------------
# Local dates
# ------------------------------------------------------------

def test_evening_storm_uses_its_local_date(monkeypatch):
    # 00:46 UTC on Mar 27 is 8:46 PM EDT on Mar 26 in Kenton, OH.
    kenton = feature(1, "Kenton", start_ms=epoch_ms(2026, 3, 27, 0, 46), lat=40.65, lon=-83.6)
    serve_pages(monkeypatch, {0: page([kenton])})

    on_26th = td.get_dat_tornadoes(date(2026, 3, 26), date(2026, 3, 26))
    on_27th = td.get_dat_tornadoes(date(2026, 3, 27), date(2026, 3, 27))

    assert list(on_26th["local_date"]) == [date(2026, 3, 26)]
    assert on_26th["tz"].iloc[0] == "America/New_York"
    assert on_27th.empty


def test_server_query_window_is_padded_around_the_local_range(monkeypatch):
    calls = serve_pages(monkeypatch, {0: page([])})

    td.get_dat_tornadoes(date(2026, 3, 26), date(2026, 3, 26))

    where = calls[0]["where"]
    assert "stormdate >= DATE '2026-03-25'" in where
    assert "stormdate < DATE '2026-03-28'" in where


# ------------------------------------------------------------
# Photos
# ------------------------------------------------------------

def test_photo_point_query_escapes_quotes(monkeypatch):
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append((url, params))
        if url.endswith("/attachments"):
            return FakeResponse({"attachmentInfos": []})
        return FakeResponse({"features": []})

    monkeypatch.setattr(td.requests, "get", fake_get)

    td.get_tornado_photos("O'Neill", 42)

    point_queries = [p for u, p in seen if u.endswith("/0/query")]
    assert point_queries[0]["where"] == "event_id = 'O''Neill'"


def test_synthetic_event_id_skips_point_query(monkeypatch):
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(url)
        return FakeResponse({"attachmentInfos": []})

    monkeypatch.setattr(td.requests, "get", fake_get)

    assert td.get_tornado_photos(f"{td.SYNTHETIC_EVENT_ID_PREFIX}42", 42) == []
    assert not any(u.endswith("/0/query") for u in seen)


# ------------------------------------------------------------
# Boundary download cache
# ------------------------------------------------------------

def test_cached_download_skips_network_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(td, "CACHE_DIR", tmp_path)
    (tmp_path / "states.zip").write_bytes(b"cached")

    def fail(*args, **kwargs):
        raise AssertionError("should not hit the network")

    monkeypatch.setattr(td.requests, "get", fail)

    path = td._cached_download("https://example.test/geo/states.zip")

    assert path.read_bytes() == b"cached"


def test_cached_download_saves_on_first_use(monkeypatch, tmp_path):
    monkeypatch.setattr(td, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(td.requests, "get", lambda *a, **k: FakeResponse(content=b"zipdata"))

    path = td._cached_download("https://example.test/geo/states.zip")

    assert path == tmp_path / "cache" / "states.zip"
    assert path.read_bytes() == b"zipdata"


# ------------------------------------------------------------
# Tornado table
# ------------------------------------------------------------

def test_tornado_table_and_detail_formatting(monkeypatch):
    kenton = feature(7, "032726-Kenton", start_ms=epoch_ms(2026, 3, 27, 0, 46))
    kenton["properties"].update({"propdamage": 800, "cropdamage": 0, "length": 0.13})
    serve_pages(monkeypatch, {0: page([kenton])})
    tornadoes = td.get_dat_tornadoes(date(2026, 3, 26), date(2026, 3, 26))
    touches = pd.DataFrame({
        "event_id": ["032726-Kenton"], "STUSPS": ["OH"],
        "NAME": ["Hardin"], "NAMELSAD": ["Hardin County"],
    })

    table = td.build_tornado_table(tornadoes, touches)
    row = table.iloc[0]
    detail = td.format_tornado_detail(row)

    assert row["Date"] == date(2026, 3, 26)
    assert row["Time"] == "8:46 PM EDT"
    assert row["Counties"] == "Hardin County, OH"
    assert row["Property Damage ($K)"] == 800
    assert detail["Start"] == "2026-03-26 8:46 PM EDT (00:46 UTC)"
    assert detail["Property Damage"] == "$800K"
    assert detail["Crop Damage"] == "$0"


def test_not_reported_sentinel_shows_as_missing(monkeypatch):
    shandon = feature(8, "Shandon OH")
    shandon["properties"].update({"propdamage": -99, "maxwind": -99, "width": -99})
    serve_pages(monkeypatch, {0: page([shandon])})
    tornadoes = td.get_dat_tornadoes(date(2026, 3, 26), date(2026, 3, 26))

    row = td.build_tornado_table(tornadoes, pd.DataFrame(
        columns=["event_id", "STUSPS", "NAME", "NAMELSAD"]
    )).iloc[0]
    detail = td.format_tornado_detail(row)

    assert pd.isna(row["Property Damage ($K)"])
    assert detail["Property Damage"] == "—"
    assert detail["Max Wind (mph)"] == "—"
    assert detail["Path Width (yd)"] == "—"
