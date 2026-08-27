"""Synthetic datasets that mimic the *shape* of the three papers' data.

The real measurement data behind the papers is either deposited on Zenodo
(Petrić et al. 2024) or available on request (the other two). These generators
produce data with the same columns, frequency, and rough statistical structure
so the reproduction pipelines run end-to-end offline. Swap ``make_*`` for a
``nextaire_tools.load_table(...)`` of the real files to reproduce the published numbers.

Nothing here requires network access or optional dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The five Graz monitoring stations used in Petrić et al. (2024) and
# Jiménez-Navarro et al. (2024).
GRAZ_STATIONS = ("Nord", "Sued", "West", "Ost", "DonBosco")

# The four Zagreb stations in Račić et al. (2026).
ZAGREB_STATIONS = ("IMI", "Siget", "ZG1", "ZG3")

# PM10-bound species measured in Račić et al. (2026).
PAHS = ("BaP", "BaA", "BbF", "BkF", "Chry", "Flu", "Pyr")
METALS = ("As", "Cd", "Pb", "Mn", "Fe", "Cu", "Zn")


def make_graz_hourly(
    n_days: int = 365,
    start: str = "2016-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Hourly, datetime-indexed frame for one Graz station (Papers 1 & 2 shape).

    Columns: pollutants (``no``, ``no2``, ``o3``, ``pm10``), ground meteorology
    (``temp``, ``rh``, ``pressure``, ``wind_speed``, ``wind_dir``, ``radiation``,
    ``precip``) and a couple of ERA5-style reanalysis variables (``blh``,
    ``u10``, ``v10``). Realistic diurnal/seasonal cycles, gaps, and spikes.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=24 * n_days, freq="h", name="timestamp")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()

    diurnal = np.sin((hour - 6) * 2 * np.pi / 24)
    seasonal = np.cos((doy - 15) * 2 * np.pi / 365)  # cold in Jan

    temp = 10 + 12 * seasonal + 6 * diurnal + rng.normal(0, 2, n)
    rh = np.clip(70 - 15 * diurnal + rng.normal(0, 8, n), 10, 100)
    pressure = 1013 + rng.normal(0, 6, n)
    wind_speed = np.clip(2.5 + 1.5 * rng.gamma(2.0, 1.0, n) / 2, 0.1, None)
    wind_dir = (180 + 90 * np.sin(doy * 2 * np.pi / 365) + rng.normal(0, 40, n)) % 360
    radiation = np.clip((diurnal + 1) * 400 * (0.5 + 0.5 * seasonal), 0, None)
    precip = rng.gamma(0.3, 1.0, n) * (rng.random(n) < 0.1)
    blh = np.clip(300 + 700 * (diurnal + 1) / 2 + rng.normal(0, 100, n), 50, None)
    theta = np.deg2rad(wind_dir)
    u10 = -wind_speed * np.sin(theta)
    v10 = -wind_speed * np.cos(theta)

    # Traffic-driven NOx: morning + evening rush, worse in winter (low BLH).
    rush = np.exp(-(((hour - 8) % 24) ** 2) / 6) + np.exp(-(((hour - 18) % 24) ** 2) / 6)
    no = np.clip(40 * rush * (1000 / blh) + rng.normal(0, 8, n), 0, None)
    no2 = np.clip(0.5 * no + 15 + 8 * diurnal + rng.normal(0, 5, n), 0, None)
    # Photochemical O3: high with radiation, anti-correlated with NO.
    o3 = np.clip(30 + 0.06 * radiation - 0.3 * no + rng.normal(0, 6, n), 0, None)
    pm10 = np.clip(
        18 + 10 * seasonal + 0.2 * no + 5 * (precip == 0) + rng.gamma(2.0, 3.0, n),
        0,
        None,
    )

    df = pd.DataFrame(
        {
            "no": no,
            "no2": no2,
            "o3": o3,
            "pm10": pm10,
            "temp": temp,
            "rh": rh,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "radiation": radiation,
            "precip": precip,
            "blh": blh,
            "u10": u10,
            "v10": v10,
        },
        index=idx,
    )

    # Inject realistic gaps and a New-Year-style fireworks PM10 spike.
    gap = rng.random(n) < 0.01
    df.loc[gap, "no2"] = np.nan
    df.loc[rng.random(n) < 0.005, "o3"] = np.nan
    nye = (idx.month == 1) & (idx.day == 1) & (idx.hour == 0)
    df.loc[nye, "pm10"] += 400.0
    return df


def make_graz_multistation(n_days: int = 365, seed: int = 0) -> dict[str, pd.DataFrame]:
    """One hourly frame per Graz station (shared index), for the lagged-feature setup."""
    return {
        st: make_graz_hourly(n_days=n_days, seed=seed + i)
        for i, st in enumerate(GRAZ_STATIONS)
    }


def make_zagreb_daily(
    n_days: int = 4 * 365,
    start: str = "2017-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Daily long-format PM10-bound PAH/metal frame for Zagreb (Paper 3 shape).

    One row per (day, station). Columns: the seven PAHs and seven metals (ng/m3),
    ``pm10``, ``no2``, daily meteorology, a heating proxy (``gas_sum``), a traffic
    proxy (``traffic``), plus temporal (``julian``, ``dow``, ``month``) and the
    station label. Concentrations are heating/traffic/temperature driven so that
    NMF and Random Forest recover an interpretable structure.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for st_i, station in enumerate(ZAGREB_STATIONS):
        doy = days.dayofyear.to_numpy()
        seasonal = np.cos((doy - 15) * 2 * np.pi / 365)  # winter high
        temp = 12 + 12 * seasonal + rng.normal(0, 3, n_days)
        radiation = np.clip(150 + 120 * -seasonal + rng.normal(0, 20, n_days), 0, None)
        rh = np.clip(75 - 10 * -seasonal + rng.normal(0, 8, n_days), 20, 100)
        wind = np.clip(2.0 + rng.gamma(2.0, 0.7, n_days), 0.2, None)
        # Heating proxy: high in winter. Traffic proxy: weekly cycle, station-dependent.
        heating = np.clip(1000 * (0.5 + 0.5 * seasonal) + rng.normal(0, 80, n_days), 0, None)
        dow = days.dayofweek.to_numpy()
        traffic = (1.0 - 0.5 * (dow >= 5)) * (1.0 + 0.3 * st_i) * (1 + rng.normal(0, 0.1, n_days))

        # Two latent sources: combustion/heating and traffic/road-dust.
        combustion = np.clip(
            heating / 1000 + 0.3 * -temp / 20 + rng.normal(0, 0.2, n_days), 0, None
        )
        road = np.clip(traffic + 0.2 * rng.normal(0, 0.3, n_days), 0, None)

        frame = {"date": days, "station": station}
        # PAHs load mostly on combustion (winter heating signal).
        for j, pah in enumerate(PAHS):
            load = 0.8 + 0.05 * j
            frame[pah] = np.clip(
                load * combustion * (2.0 + j * 0.2) + 0.1 * road + rng.gamma(1.5, 0.05, n_days),
                1e-4,
                None,
            )
        # Metals load mostly on traffic/road dust and are PM10/NO2-correlated.
        for j, metal in enumerate(METALS):
            frame[metal] = np.clip(
                (1.0 + 0.1 * j) * road * (1.5 + j * 0.3) + 0.2 * combustion
                + rng.gamma(1.5, 0.2, n_days),
                1e-3,
                None,
            )
        pm10 = np.clip(20 + 12 * seasonal + 8 * road + rng.gamma(2.0, 3.0, n_days), 0, None)
        no2 = np.clip(15 + 10 * road + 5 * -temp / 20 + rng.normal(0, 4, n_days), 0, None)
        frame.update(
            {
                "pm10": pm10,
                "no2": no2,
                "temp": temp,
                "temp_min": temp - rng.uniform(2, 6, n_days),
                "temp_max": temp + rng.uniform(2, 6, n_days),
                "radiation": radiation,
                "rh": rh,
                "wind_speed": wind,
                "gas_sum": heating,
                "traffic": traffic,
                "julian": (days - pd.Timestamp("1970-01-01")).days.to_numpy(),
                "dow": dow,
                "month": days.month.to_numpy(),
            }
        )
        rows.append(pd.DataFrame(frame))

    out = pd.concat(rows, ignore_index=True)
    # Exclude the COVID lockdown window, as the paper does.
    lockdown = (out["date"] >= "2020-03-15") & (out["date"] <= "2020-05-11")
    return out.loc[~lockdown].reset_index(drop=True)
