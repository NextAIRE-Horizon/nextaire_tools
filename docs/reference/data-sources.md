# Copernicus data sources

A reference for the reanalysis stores `nextaire_tools` downloads from: what each one
serves, its API endpoint, the variables you are likely to want, and the
conventions and units you must get right. For the code-level workflow, see the
[Copernicus extractors guide](../user-guide/extractors.md); for signatures, the
[`nextaire_tools.extractors` API](../api/extractors.md).

## The two stores (plus a third door)

Copernicus splits reanalysis across two data stores, each with its own endpoint,
registration and Personal Access Token:

- **CDS — Climate Data Store.** ECMWF's *physical* atmosphere/land reanalysis:
  ERA5 and ERA5-Land. Served by [`ERA5Extractor`](../api/extractors.md) and
  [`ERA5LandExtractor`](../api/extractors.md).
- **ADS — Atmosphere Data Store.** *Atmospheric-composition* reanalysis and
  forecasts: CAMS. Served by [`CAMSExtractor`](../api/extractors.md).

A third family — CLMS land-cover *rasters* — is not on either `cdsapi` store and
goes through WEkEO/HDA instead; see [Land and CLMS](#land-and-clms) below.

| Store | API endpoint | Example datasets | Typical variables | Temporal resolution |
|---|---|---|---|---|
| **CDS** (Climate Data Store) | `https://cds.climate.copernicus.eu/api` | `reanalysis-era5-single-levels`, `reanalysis-era5-land` | 2 m temperature, 10 m u/v wind, surface pressure, total precipitation, boundary-layer height, net solar radiation; (Land:) soil moisture, LAI, skin temperature, snow | **Hourly** (both ERA5 and ERA5-Land) |
| **ADS** (Atmosphere Data Store) | `https://ads.atmosphere.copernicus.eu/api` | `cams-global-reanalysis-eac4`, `cams-global-atmospheric-composition-forecasts` | PM2.5, PM10, NO₂, O₃, CO, SO₂ and other reactive gases and aerosols | **3-hourly** (EAC4 reanalysis); forecasts by lead time |

!!! note "One account, two stores"
    You register and accept dataset licences separately on the CDS and ADS web
    portals. Each extractor already points its `url` at the right endpoint, but
    the token still comes from `key=` / `CDSAPI_KEY` / `~/.cdsapirc`, so supply
    the token that matches the store you are querying. See
    [Credentials](../user-guide/extractors.md#1-credentials).

## CDS — ERA5 and ERA5-Land

- **ERA5 single levels** (`reanalysis-era5-single-levels`) is ECMWF's global
  atmospheric reanalysis on a regular lat/lon grid (~31 km), **hourly** from
  1940 to near-present. It is the standard source of meteorological predictors
  for air-quality models: wind, temperature, pressure, precipitation,
  boundary-layer height and radiation. `ERA5Extractor.build_request` accepts a
  `product_type` (default `"reanalysis"`) and expands `start`/`end` into CDS
  `year`/`month`/`day` lists.
- **ERA5-Land** (`reanalysis-era5-land`) replays the land component at higher
  (~9 km) resolution, also **hourly**. Use it for land-surface predictors — soil
  moisture, skin temperature, leaf-area index, snow. Its request has **no**
  `product_type` key.

Overviews and full variable tables:

- ERA5 single levels: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>
- ERA5-Land: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land>
- CDS API setup: <https://cds.climate.copernicus.eu/how-to-api>

## ADS — CAMS

- **CAMS global reanalysis EAC4** (`cams-global-reanalysis-eac4`) provides
  global atmospheric-composition fields — particulate matter and reactive gases
  — at **3-hourly** steps (`00:00, 03:00, …, 21:00`). `CAMSExtractor` targets it
  by default and, unlike ERA5, uses a single `date` **range string**
  (`"YYYY-MM-DD/YYYY-MM-DD"`) rather than separate year/month/day lists.
- **CAMS forecasts** (`cams-global-atmospheric-composition-forecasts`) are the
  near-real-time counterpart. They reuse the EAC4 building blocks but require
  `leadtime_hour` and `type` keys; build a base request, extend it, and pass the
  dataset id explicitly (see
  [Requesting CAMS forecasts](../user-guide/extractors.md#requesting-cams-forecasts)).

Overviews and full variable tables:

- CAMS EAC4 reanalysis: <https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4>
- CAMS composition forecasts: <https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts>
- ADS API setup: <https://ads.atmosphere.copernicus.eu/how-to-api>

## Land and CLMS

The Copernicus Land Monitoring Service (**CLMS**) publishes land-*cover* and
vegetation **raster** products — CORINE land cover, high-resolution layers
(imperviousness, tree-cover density), NDVI and LAI time series. These are **not**
available through `cdsapi`; they are distributed via **WEkEO** and its
Harmonised Data Access (**HDA**) API.

- CLMS portal: <https://land.copernicus.eu/>
- WEkEO / HDA: <https://www.wekeo.eu/>

!!! tip "OSM land-use regression as an alternative"
    For per-station *land-use* features (fraction of road, building or green
    space around each site), the OpenStreetMap land-use-regression (LUR)
    approach in `notebooks/LurExtractor.ipynb` (`osmnx`/`geopandas`, installed
    via the `"geo"` extra) is a practical alternative that avoids any Copernicus
    raster download. See the
    [extractors guide](../user-guide/extractors.md#7-land-cover-and-land-use-features).

## The `area` bounding box: `[North, West, South, East]`

Every Copernicus request takes an `area` box in **decimal degrees**, ordered
**`[North, West, South, East]`** — latitude bounds first, then longitude bounds:

```python
area = [49.1, 9.5, 46.3, 17.2]
#       ^N    ^W   ^S    ^E
```

This is a different order from the `(min_lon, min_lat, max_lon, max_lat)`
convention many GIS tools use, and getting it wrong silently downloads the wrong
region. `nextaire_tools` validates the box (four floats) and can compute one for you from
your stations with `CopernicusExtractor.expand_area(stations, margin=0.5)`.

!!! danger "Checkpoint — box order and the date grid"
    - `area` is `[N, W, S, E]`, not `[W, S, E, N]` or `(lon, lat, …)`.
    - For CDS ERA5/ERA5-Land, `start`/`end` become independent
      `year`/`month`/`day` lists whose **Cartesian product** is retrieved. A
      range crossing a month boundary pulls extra dates — request one month at a
      time for an exact window. CAMS avoids this by using a single date-range
      string.

## Nearest-neighbour sampling

`nextaire_tools` samples reanalysis at your stations by **nearest neighbour**: each
station's `(lon, lat)` is snapped to the closest grid cell
(`xarray`'s `.sel(..., method="nearest")`). Consequences to keep in mind:

- No interpolation is performed — the value is the containing grid cell's value.
- Two stations closer together than the grid spacing (~31 km for ERA5, ~9 km for
  ERA5-Land, coarser for CAMS) can be assigned the **same** cell and identical
  values.
- A station just outside your `area` box still snaps to the nearest *in-box*
  cell, so pad the box (`expand_area`'s `margin`) to keep the true nearest cell
  inside it.

## Units caveats

Reanalysis fields are in **SI units**, which are rarely the units you report in.
Always check the dataset's variable table before combining fields or comparing
to observations.

- **ERA5 / ERA5-Land.** Temperatures in **kelvin** (subtract 273.15 for °C),
  pressure in **Pa**, winds in **m s⁻¹**, boundary-layer height in **m**.
  `total_precipitation` and the radiation fields are **accumulations** (metres of
  water equivalent; J m⁻²) over the model step, not instantaneous rates — treat
  them accordingly before differencing or resampling.
- **CAMS EAC4.** Many gas-phase species (NO₂, O₃, CO, SO₂) are provided as
  **mass mixing ratios** (kg kg⁻¹), *not* concentrations. Converting to the
  µg m⁻³ you compare against ground observations requires the local **air
  density** (from temperature and pressure). Particulate-matter fields are
  typically supplied as **mass concentrations** (kg m⁻³) instead. Because the
  representation differs by variable, confirm each one in the EAC4
  documentation rather than assuming a single conversion.

!!! warning "Do not compare raw CAMS gas fields to station µg/m³"
    A CAMS NO₂ mass mixing ratio and a station's NO₂ concentration are different
    physical quantities. Convert (or at minimum standardise both) before using
    one to validate or bias-correct the other.

## See also

- [Copernicus extractors guide](../user-guide/extractors.md) — the end-to-end
  download workflow.
- [`nextaire_tools.extractors` API](../api/extractors.md) — `ERA5Extractor`,
  `CAMSExtractor`, `ERA5LandExtractor`, `load_stations`, and the GRIB helpers.
