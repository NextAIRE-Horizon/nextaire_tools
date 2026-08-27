# Copernicus extractors

`nextaire_tools.extractors` downloads meteorological and atmospheric-composition
reanalysis from the Copernicus data stores and samples the grid at your
monitoring stations, returning one tidy, timestamp-indexed `DataFrame` per
station. Those frames become the *predictor* features you join to your observed
pollutant columns (`no2`, `o3`, `pm10`) before modeling.

The extractors are an optional feature. Install the extra dependencies
(`cdsapi`, `xarray`, `cfgrib`) with:

```bash
pip install "nextaire_tools[extract]"
```

!!! note "What you get"
    Three thin subclasses of `CopernicusExtractor` —
    [`ERA5Extractor`](../api/extractors.md), [`CAMSExtractor`](../api/extractors.md)
    and [`ERA5LandExtractor`](../api/extractors.md) — plus station helpers
    (`load_stations`, `dms_to_dd`) and the lower-level GRIB pipeline
    (`extract_archive`, `open_grib_datasets`, `sample_at_points`,
    `merge_station_frames`). See the [API reference](../api/extractors.md) for
    full signatures and the [data-sources reference](../reference/data-sources.md)
    for what each store contains.

## 1. Credentials

Every download goes through a Copernicus data store that requires a **Personal
Access Token (PAT)**. There are two separate stores:

- **CDS** (Climate Data Store) — `https://cds.climate.copernicus.eu/api` — serves
  ERA5 and ERA5-Land.
- **ADS** (Atmosphere Data Store) — `https://ads.atmosphere.copernicus.eu/api` —
  serves CAMS.

You must register on each store, accept the licence for each dataset you intend
to download, and copy your PAT from your account page. The token is read, in
order of precedence, from:

1. the `key=` argument to the extractor constructor,
2. the `CDSAPI_KEY` environment variable,
3. a `~/.cdsapirc` file (read by `cdsapi` itself).

=== "`~/.cdsapirc` (CDS / ERA5)"

    ```
    url: https://cds.climate.copernicus.eu/api
    key: <your-CDS-personal-access-token>
    ```

=== "`~/.cdsapirc` (ADS / CAMS)"

    ```
    url: https://ads.atmosphere.copernicus.eu/api
    key: <your-ADS-personal-access-token>
    ```

=== "Environment variable"

    ```bash
    export CDSAPI_KEY="<your-personal-access-token>"
    ```

=== "In code"

    ```python
    from nextaire_tools.extractors import ERA5Extractor

    extractor = ERA5Extractor(key="<your-personal-access-token>")
    ```

!!! warning "CDS and ADS are different stores"
    Each extractor already points its `url` at the correct store, but the
    **token** still falls back to `CDSAPI_KEY` / `~/.cdsapirc`. If your
    `~/.cdsapirc` holds a CDS token and you run `CAMSExtractor`, authentication
    will fail against the ADS endpoint. When juggling both stores, pass the
    matching token explicitly with `key=` (or keep the ADS token in
    `~/.cdsapirc` while running CAMS).

If the client cannot be constructed, `nextaire_tools` raises a `CredentialsError` whose
message repeats the expected `~/.cdsapirc` layout and links to the setup guide.
Catch it to fail fast with a clear message:

```python
from nextaire_tools.exceptions import CredentialsError
from nextaire_tools.extractors import ERA5Extractor

try:
    extractor = ERA5Extractor()
    frames = extractor.extract_to_frames(stations=stations, area=area,
                                          start="2026-01-01", end="2026-01-02")
except CredentialsError as exc:
    raise SystemExit(f"Set up your CDS token first:\n{exc}")
```

## 2. Station metadata

Sampling needs station coordinates in a `DataFrame` with exactly the columns
`station_name`, `station_lon` and `station_lat` (decimal degrees).
`load_stations` produces exactly that from a CSV, Excel, or Parquet file, and it
copes with the two things real station spreadsheets get wrong:

- **DMS coordinates.** Cells such as `"15°58'44\"E"` or `"45°48'36\"N"` are
  converted to decimal degrees; `S`/`W` (or a leading minus) yield negative
  values. Decimal cells pass through unchanged.
- **Stray header whitespace.** Column names are matched *after* stripping
  surrounding spaces, so a header like `"Measuring station "` (note the trailing
  space, common in the source spreadsheets) still resolves to the default
  `name_col="Measuring station"`.

```python
from nextaire_tools.extractors import load_stations

stations = load_stations(
    "data/Coordinates.xlsx",
    name_col="Measuring station",   # trailing spaces are tolerated
    lon_col="Longitude",
    lat_col="Latitude",
)
list(stations.columns)
# ['station_name', 'station_lon', 'station_lat']
```

If any requested column is missing, `load_stations` raises a `SchemaError` that
lists the columns it *did* find — handy for spotting a typo in `lon_col`/`lat_col`.

For a single value, use `dms_to_dd` directly:

```python
from nextaire_tools.extractors import dms_to_dd

dms_to_dd("46°18'27\"N")   # 46.3075
dms_to_dd("9°31'48\"W")    # -9.53
dms_to_dd(46.3075)          # 46.3075 (non-strings pass through unchanged)
```

!!! danger "Checkpoint — coordinate parsing and area order"
    - Coordinates may arrive as DMS *strings*; `load_stations`/`dms_to_dd`
      handle both DMS and decimal, so never eyeball-convert by hand.
    - The Copernicus `area` bounding box is **`[North, West, South, East]`** — a
      different order from many GIS tools. Get it wrong and you silently
      download the antipodes. See the [convention note](../reference/data-sources.md).
    - Grid sampling is **nearest-neighbour** to the reanalysis grid: a station is
      snapped to the closest grid cell, so two nearby stations can share a value.

## 3. The high-level path: `extract_to_frames`

`extract_to_frames` runs the whole pipeline — build the request, download the
archive to a temporary directory, extract the GRIB, sample at every station, and
merge — returning `{station_name: DataFrame}`. It is the method you will use
most.

```python
from nextaire_tools.extractors import ERA5Extractor, load_stations

stations = load_stations("data/Coordinates.xlsx")

extractor = ERA5Extractor()
frames = extractor.extract_to_frames(
    stations=stations,
    area=[49.1, 9.5, 46.3, 17.2],   # [North, West, South, East]
    start="2026-01-01",
    end="2026-01-31",
    save_dir="data/era5",           # optional: one <station>.csv per station
)

zagreb = frames["Zagreb-1"]
zagreb.head()      # timestamp-indexed, one column per GRIB variable (t2m, u10, ...)
```

Everything after `stations=` is a **build keyword** forwarded to
`build_request` (see below): `area`, `start`, `end`, and the optional
`variables`, `times`, `data_format`. When `save_dir` is given, each station
frame is written as `<safe_station_name>.csv` (names sanitised with
`safe_filename`).

The returned frames are indexed by `timestamp` with one column per GRIB data
variable — for ERA5 these are the short cfgrib names (e.g. `t2m` for 2 m
temperature, `u10`/`v10` for the 10 m wind components, `blh` for the boundary
layer height). Join them onto your pollutant observations on the shared
timestamp index, then hand the combined frame to the
[preprocessing pipeline](pipelines.md) and [models](modeling.md).

!!! tip "Let the stations define the box"
    `CopernicusExtractor.expand_area` computes an `[N, W, S, E]` box that covers
    all your stations plus a margin, so you don't hand-pick coordinates:

    ```python
    area = ERA5Extractor.expand_area(stations, margin=0.5)  # degrees of padding
    frames = extractor.extract_to_frames(stations=stations, area=area,
                                         start="2026-01-01", end="2026-01-31")
    ```

### Passing an explicit request

If you already hold a request dictionary, pass it as the first positional
argument instead of build keywords. Supplying **both** raises a
`ConfigurationError`:

```python
request = extractor.build_request(area=area, start="2026-01-01", end="2026-01-02")
frames = extractor.extract_to_frames(request, stations=stations)   # OK
# extractor.extract_to_frames(request, stations=stations, area=area)  # ConfigurationError
```

## 4. The three extractors

All three share the constructor and `extract_to_frames`; they differ only in the
store they target, their default dataset and variables, and the shape of
`build_request`.

=== "ERA5 (CDS, hourly)"

    ```python
    from nextaire_tools.extractors import ERA5Extractor

    ex = ERA5Extractor()
    req = ex.build_request(
        area=[49.1, 9.5, 46.3, 17.2],   # [N, W, S, E]
        start="2026-01-01",
        end="2026-01-31",
        # variables=[...],              # defaults to ERA5Extractor.DEFAULT_VARIABLES
        # times=[f"{h:02d}:00" for h in range(24)],  # defaults to every hour
        product_type="reanalysis",      # ERA5 only
    )
    ```

    Dataset `reanalysis-era5-single-levels`; **hourly**. `build_request`
    accepts `product_type` and expands `start`/`end` into CDS
    `year`/`month`/`day` lists.

=== "CAMS EAC4 (ADS, 3-hourly)"

    ```python
    from nextaire_tools.extractors import CAMSExtractor

    ex = CAMSExtractor()
    req = ex.build_request(
        area=[49.1, 9.5, 46.3, 17.2],
        start="2023-01-01",
        end="2023-01-07",
        # times default to the 3-hourly EAC4 steps: 00:00, 03:00, ..., 21:00
    )
    req["date"]   # '2023-01-01/2023-01-07'  (a single date-range string)
    ```

    Dataset `cams-global-reanalysis-eac4`; **3-hourly**. Uses a single `date`
    range string, not `year`/`month`/`day`, and has **no** `product_type`.

=== "ERA5-Land (CDS, hourly)"

    ```python
    from nextaire_tools.extractors import ERA5LandExtractor

    ex = ERA5LandExtractor()
    req = ex.build_request(
        area=[49.1, 9.5, 46.3, 17.2],
        start="2026-01-01",
        end="2026-01-31",
    )
    "product_type" in req   # False
    ```

    Dataset `reanalysis-era5-land`; **hourly**, ~9 km resolution. Same request
    shape as ERA5 but with **no** `product_type` key.

!!! danger "Checkpoint — know your store's cadence"
    - **ERA5** is hourly; **CAMS EAC4** is 3-hourly; **ERA5-Land** is hourly.
    - Passing `product_type=` to ERA5-Land or CAMS is a mistake — those datasets
      have no product-type dimension, and only `ERA5Extractor.build_request`
      accepts it.
    - A 3-hourly CAMS frame will not align row-for-row with hourly pollutant
      observations. Reindex/resample one to the other (e.g. forward-fill CAMS to
      hourly, or aggregate pollutants to 3-hourly) *before* joining, or you will
      introduce `NaN`s that a later [`Scaler`](scaling.md) propagates.

### Requesting CAMS forecasts

The near-real-time forecast dataset
`cams-global-atmospheric-composition-forecasts` reuses the EAC4 building blocks
but additionally needs `leadtime_hour` and `type`. Extend the request and pass
the dataset id explicitly:

```python
ex = CAMSExtractor()
request = ex.build_request(area=area, start="2026-07-01", end="2026-07-02")
request["leadtime_hour"] = ["0", "12", "24"]
request["type"] = "forecast"
ex.retrieve(request, "cams_fc.zip",
            dataset="cams-global-atmospheric-composition-forecasts")
```

## 5. Large requests: run from a script, not interactively

CDS/ADS requests are **queued server-side** and can take minutes to hours: a
single month of hourly ERA5 across a handful of variables is already a
multi-gigabyte GRIB download. Do not block a notebook cell waiting on it.

!!! warning "Checkpoint — offload big downloads"
    Run extractions from a standalone script or a background job, not an
    interactive cell you have to babysit. The repository ships a worked example
    at `scripts/Era5Extractor.py` you can adapt:

    ```bash
    python scripts/Era5Extractor.py
    ```

    Slice long date ranges into per-month requests. Note that
    `start`/`end` are expanded into independent CDS `year`/`month`/`day` lists,
    so a range spanning a month boundary (e.g. `2020-12-30`..`2021-01-02`)
    downloads the full Cartesian product of those lists — request one month at a
    time when you need an exact window.

## 6. Lower-level GRIB pipeline (advanced)

`extract_to_frames` is a thin wrapper over four composable helpers. Call them
directly when you want to keep the raw GRIB, inspect the `xarray` datasets, or
sample archives you downloaded by other means.

```python
from nextaire_tools.extractors import (
    extract_archive, open_grib_datasets, sample_at_points, merge_station_frames,
)

# 1. Download the archive yourself (returns the target Path).
extractor = ERA5Extractor()
request = extractor.build_request(area=area, start="2026-01-01", end="2026-01-02")
archive = extractor.retrieve(request, "data/era5.zip")

# 2. Unzip and locate the GRIB messages.
grib_files = extract_archive(archive, "data/era5_extracted")

# 3. Open them — one xarray.Dataset per typeOfLevel grouping.
datasets = open_grib_datasets(grib_files)

# 4. Sample each dataset at the stations, then merge per station.
per_dataset = [sample_at_points(ds, stations) for ds in datasets]
frames = merge_station_frames(per_dataset)   # {station_name: DataFrame}
```

Notes on behaviour:

- `retrieve(request, target, dataset=None)` downloads to `target` (creating
  parent directories) and returns the `Path`; it raises `ExtractionError` on a
  failed download.
- `extract_archive` returns the sorted `.grib`/`.grb`/`.grib2` files inside the
  ZIP, and passes a raw GRIB file straight through if the input is not a ZIP.
- `open_grib_datasets` returns **one dataset per `typeOfLevel` group**, so a
  single GRIB file can yield several datasets — that is why sampling maps over a
  list and `merge_station_frames` concatenates the per-group frames column-wise
  (dropping duplicate columns).
- `sample_at_points` auto-detects the `longitude`/`lat` coordinate names,
  normalises the time coordinate to `timestamp`, and returns
  `{station_name: DataFrame}`.

## 7. Land cover and land-use features

ERA5-Land covers land-*surface* meteorology (soil moisture, skin temperature,
LAI, snow), but dedicated Copernicus Land Monitoring Service (**CLMS**) raster
products — land cover, high-resolution NDVI, imperviousness — are **not**
available through `cdsapi`. They are distributed via **WEkEO** and its
Harmonised Data Access (**HDA**) API. See the
[data-sources reference](../reference/data-sources.md#land-and-clms) for links.

!!! note "OSM land-use regression as an alternative"
    For per-station *land-use* features — the fraction of road, building or
    green space in a radius around each site — the OpenStreetMap land-use
    regression (LUR) approach in the project's `notebooks/LurExtractor.ipynb`
    (using `osmnx`/`geopandas`, installed via the `"geo"` extra) is a practical
    alternative that needs no Copernicus raster download.

## See also

- [Data-sources reference](../reference/data-sources.md) — what each store
  contains, the `[N, W, S, E]` convention, and units caveats.
- [Loading data](loading-data.md) — reading the CSVs `save_dir` writes back in.
- [Pipelines](pipelines.md) and [Modeling](modeling.md) — turning the extracted
  frames into leakage-free predictors and models.
- [`nextaire_tools.extractors` API](../api/extractors.md) — full signatures.
