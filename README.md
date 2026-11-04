# Tornado Counter

Counts surveyed NWS tornadoes per U.S. state for a date range, using
the National Weather Service [Damage Assessment Toolkit][dat] (DAT).
A Streamlit app shows the counts as a table and a map, and lets you
drill into a state's individual tornadoes, including survey narratives
and photos.

## Running it

```sh
pip install -r requirements.txt
streamlit run app.py
```

Command-line counts (written to `tornado_counts.csv`):

```sh
python tornado_counts.py --start 2026-01-01 --end 2026-06-30
```

Tests:

```sh
pip install -r requirements-dev.txt
pytest
```

## Using the app

- **Date range** presets (year to date, last 7/30 days, last year) or
  a custom range. The page loads year to date automatically.
- **EF rating** filter applies to everything on the page.
- **Map view** toggles between per-state counts and each tornado's
  track, colored by EF rating.
- Click a state in the table or on the map to list its tornadoes.
  Column headers sort; selecting a row shows the detail card,
  narrative and survey photos.
- Both tables download as CSV.
- The URL reflects the current view (dates, filters, map view and
  selected state), so it can be bookmarked or shared.

## How the data is interpreted

- **Multi-state tracks count once per state they touch.** A tornado
  that crosses from Indiana into Ohio counts for both, so the sum of
  the state counts can exceed the number of distinct tornadoes.
- **Dates are local.** DAT stores times in UTC; the app converts each
  tornado to the time zone where it started, matching how NWS offices
  date tornadoes. An evening storm is listed on its local date, not the
  next UTC day.
- **Damage is in thousands of dollars**, as labeled in the DAT viewer
  ("$k"). Some offices appear to enter whole dollars instead, so
  unusually large figures may be off by a factor of 1,000. `-99`, DAT's
  "not reported" value, is shown as blank.
- State and county boundaries come from the Census Bureau's 2025
  cartographic boundary files, cached under `~/.cache/tornadocounter`
  after the first download.

## Comparing with NWS office tallies

DAT holds preliminary survey data and is revised as surveys are
updated. An office's published list (for example a yearly totals PDF)
is a snapshot that may predate or postdate DAT edits, may record a
track's length or rating slightly differently, and lists a tornado
once per county it crosses while still counting it once in its total.
Expect small differences, and compare individual tornadoes by date,
county and office rather than totals alone.

[dat]: https://apps.dat.noaa.gov/StormDamage/DamageViewer/
