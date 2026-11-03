"""Generative model for SD-147, Metro Manila urban air quality, calendar year 2024.

Hourly concentrations are lognormal around a station baseline, pushed by a shared
traffic wave, a dry-season cycle, wind, and rain. PM10 is PM2.5 plus coarse dust,
so it can never fall below PM2.5. Ozone is photochemical and is titrated by NO2.
Four deliberate edge cases are written on top of that process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 147
YEAR = 2024

POLLUTANTS = [
    {
        "pollutant_id": "PM25",
        "name": "Fine particulate matter",
        "formula": "PM2.5",
        "unit": "µg/m³",
        "primary_source": "Traffic, combustion, secondary aerosol",
        "drives_aqi": True,
    },
    {
        "pollutant_id": "PM10",
        "name": "Inhalable particulate matter",
        "formula": "PM10",
        "unit": "µg/m³",
        "primary_source": "Road dust, construction, sea salt, PM2.5",
        "drives_aqi": True,
    },
    {
        "pollutant_id": "NO2",
        "name": "Nitrogen dioxide",
        "formula": "NO2",
        "unit": "µg/m³",
        "primary_source": "Vehicle exhaust",
        "drives_aqi": True,
    },
    {
        "pollutant_id": "O3",
        "name": "Ozone",
        "formula": "O3",
        "unit": "µg/m³",
        "primary_source": "Photochemistry, suppressed by fresh NO",
        "drives_aqi": True,
    },
    {
        "pollutant_id": "SO2",
        "name": "Sulfur dioxide",
        "formula": "SO2",
        "unit": "µg/m³",
        "primary_source": "Port fuel, industry, some diesel",
        "drives_aqi": True,
    },
    {
        "pollutant_id": "CO",
        "name": "Carbon monoxide",
        "formula": "CO",
        "unit": "mg/m³",
        "primary_source": "Incomplete combustion, gasoline traffic",
        "drives_aqi": True,
    },
]

# Geometric-mean baselines a calm hour would sit near, before diurnal and weather.
# coarse_gm is the PM10 − PM2.5 dust term, not a concentration of PM10 itself.
STATIONS = [
    dict(station_id="STA-EDSA", name="EDSA–Guadalupe", city="Makati", district="Guadalupe", station_type="roadside", lat=14.5667, lon=121.0456, elev=18, commissioned="2016-03-01", pm25=33, no2=50, so2=6.0, co=1.25, o3=32, coarse=24, spike=13.0),
    dict(station_id="STA-QCH", name="Quezon City Hall", city="Quezon City", district="Diliman", station_type="urban background", lat=14.6488, lon=121.0509, elev=32, commissioned="2014-07-15", pm25=22, no2=28, so2=4.5, co=0.62, o3=44, coarse=14, spike=8.5),
    dict(station_id="STA-MNL", name="Manila City Hall", city="Manila", district="Ermita", station_type="urban center", lat=14.5896, lon=120.9816, elev=10, commissioned="2013-01-20", pm25=29, no2=38, so2=7.5, co=0.88, o3=35, coarse=18, spike=11.0),
    dict(station_id="STA-PRT", name="Manila Port", city="Manila", district="Tondo", station_type="industrial port", lat=14.5831, lon=120.9614, elev=5, commissioned="2015-11-02", pm25=31, no2=36, so2=18.0, co=0.78, o3=30, coarse=28, spike=6.5),
    dict(station_id="STA-MKT", name="Makati CBD", city="Makati", district="Ayala", station_type="commercial roadside", lat=14.5547, lon=121.0244, elev=18, commissioned="2017-05-08", pm25=26, no2=44, so2=5.0, co=1.05, o3=34, coarse=16, spike=10.0),
    dict(station_id="STA-PSG", name="Pasig Rotonda", city="Pasig", district="Caniogan", station_type="mixed roadside", lat=14.5758, lon=121.0851, elev=16, commissioned="2018-02-12", pm25=28, no2=40, so2=5.5, co=0.92, o3=34, coarse=17, spike=9.5),
    dict(station_id="STA-MRK", name="Marikina Riverpark", city="Marikina", district="Santa Elena", station_type="valley residential", lat=14.6507, lon=121.1029, elev=14, commissioned="2018-09-01", pm25=30, no2=27, so2=4.0, co=0.60, o3=40, coarse=15, spike=9.0),
    dict(station_id="STA-LPI", name="Las Piñas Coastal", city="Las Piñas", district="Pulanglupa", station_type="coastal background", lat=14.4445, lon=120.9939, elev=8, commissioned="2019-04-18", pm25=16, no2=18, so2=3.8, co=0.38, o3=48, coarse=13, spike=5.0),
    dict(station_id="STA-VAL", name="Valenzuela Industrial", city="Valenzuela", district="Karuhatan", station_type="industrial", lat=14.7004, lon=120.9830, elev=12, commissioned="2015-06-22", pm25=35, no2=41, so2=15.0, co=0.84, o3=28, coarse=30, spike=7.0),
    dict(station_id="STA-ANT", name="Antipolo Upland", city="Antipolo", district="San Roque", station_type="upland background", lat=14.6255, lon=121.2142, elev=210, commissioned="2019-10-09", pm25=11, no2=10, so2=2.6, co=0.20, o3=54, coarse=7, spike=3.2),
    dict(station_id="STA-BGC", name="Bonifacio Global City", city="Taguig", district="Fort Bonifacio", station_type="commercial", lat=14.5515, lon=121.0496, elev=16, commissioned="2020-01-15", pm25=19, no2=25, so2=3.8, co=0.50, o3=46, coarse=10, spike=8.0),
    dict(station_id="STA-NAIA", name="NAIA Terminal", city="Pasay", district="Villamor", station_type="airport roadside", lat=14.5086, lon=121.0198, elev=15, commissioned="2016-08-30", pm25=25, no2=40, so2=6.5, co=0.98, o3=32, coarse=17, spike=7.5),
]

# Dashboard sub-index breakpoints. PM uses the US EPA 24-hour table.
# Gases use the 1-hour (NO2, SO2) or 8-hour (O3, CO) tables, converted to µg/m³
# or mg/m³ at 25°C. Applied here to hourly values as a nowcast, not a regulatory AQI.
BREAKPOINTS = {
    "PM25": [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    "PM10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
    "NO2": [(0, 100, 0, 50), (101, 188, 51, 100), (189, 677, 101, 150), (678, 1220, 151, 200), (1221, 2348, 201, 300)],
    "O3": [(0, 106, 0, 50), (108, 137, 51, 100), (139, 167, 101, 150), (169, 206, 151, 200), (208, 392, 201, 300)],
    "SO2": [(0, 92, 0, 50), (93, 197, 51, 100), (198, 485, 101, 150), (486, 797, 151, 200)],
    "CO": [(0, 5.0, 0, 50), (5.1, 10.8, 51, 100), (10.9, 14.2, 101, 150), (14.3, 17.7, 151, 200), (17.8, 34.9, 201, 300)],
}

AQI_BINS = [0, 51, 101, 151, 201, 301, 501]
AQI_LABELS = ["Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]

# Fixed PH dates that pull traffic down. Holy Week is damped further in code.
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-02-10", "2024-02-25", "2024-03-28", "2024-03-29",
    "2024-04-09", "2024-05-01", "2024-06-12", "2024-08-21", "2024-08-26",
    "2024-11-01", "2024-11-02", "2024-11-30", "2024-12-08", "2024-12-24",
    "2024-12-25", "2024-12-30", "2024-12-31",
])

FIREWORKS = [
    # New Year's morning, the tail of midnight burning
    ("2024-01-01 00:00", "2024-01-01 05:00", 1.00),
    # Chinese New Year evening, smaller and shorter
    ("2024-02-10 19:00", "2024-02-10 23:00", 0.42),
    # New Year's Eve ramp at the other end of the year
    ("2024-12-31 18:00", "2024-12-31 23:00", 0.85),
]
CONSTRUCTION_STATION = "STA-VAL"
CONSTRUCTION_START = pd.Timestamp("2024-03-04")
CONSTRUCTION_END = pd.Timestamp("2024-03-15 23:00")
FAULT_STATION = "STA-EDSA"
FAULT_START = pd.Timestamp("2024-08-12")
FAULT_END = pd.Timestamp("2024-08-18 23:00")
TYPHOON_START = pd.Timestamp("2024-09-15 06:00")
TYPHOON_END = pd.Timestamp("2024-09-17 18:00")
GAP_RATE = 0.004


def ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    shock = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = shock[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + shock[i]
    return out


def subindex(values: np.ndarray, pollutant_id: str) -> np.ndarray:
    out = np.zeros(values.shape, dtype=float)
    finite = np.isfinite(values)
    for lo, hi, aqi_lo, aqi_hi in BREAKPOINTS[pollutant_id]:
        mask = finite & (values >= lo) & (values <= hi)
        span = hi - lo
        out[mask] = aqi_lo + (aqi_hi - aqi_lo) * (values[mask] - lo) / span
    last_hi = BREAKPOINTS[pollutant_id][-1][1]
    out[finite & (values > last_hi)] = 500
    out[~finite] = np.nan
    return out


def aqi_label(aqi: float) -> str:
    if not np.isfinite(aqi):
        return "No reading"
    for upper, label in zip(AQI_BINS[1:], AQI_LABELS):
        if aqi < upper:
            return label
    return AQI_LABELS[-1]


def _clock(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy().astype(float)
    dow = index.dayofweek.to_numpy()
    morning = np.exp(-0.5 * ((hour - 8.0) / 1.45) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.5) / 1.9) ** 2)
    traffic = 0.16 + 1.00 * morning + 0.82 * evening
    # Sunday quiet, Saturday a step down, public holidays quieter still.
    dow_mult = np.ones(len(index))
    dow_mult[dow == 5] = 0.84
    dow_mult[dow == 6] = 0.68
    holiday = np.isin(index.normalize(), HOLIDAYS)
    dow_mult[holiday] = 0.55
    holy = (index >= "2024-03-24") & (index <= "2024-03-31 23:00")
    dow_mult[holy] = np.minimum(dow_mult[holy], 0.62)
    traffic = traffic * dow_mult
    photo = np.exp(-0.5 * ((hour - 14.0) / 2.15) ** 2)
    # Dry-season cosine peaks in early February (Amihan, little rain).
    season = np.cos(2 * np.pi * (doy - 40) / 365.25)
    wet_season = (doy >= 152) & (doy <= 334)
    return {
        "hour": hour,
        "doy": doy,
        "dow": dow,
        "traffic": traffic,
        "photo": photo,
        "season": season,
        "wet_season": wet_season,
        "holiday": holiday,
    }


def _weather(index: pd.DatetimeIndex, clock: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = len(index)
    season = clock["season"]
    hour = clock["hour"]
    wet = clock["wet_season"]
    temp_ar = ar1(n, 0.94, 0.35, rng)
    # April–May is the hot season. season peaks in February, so shift the heat wave.
    heat = np.cos(2 * np.pi * (clock["doy"] - 120) / 365.25)
    diurnal_temp = 3.6 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = 27.6 + 1.5 * heat + diurnal_temp + temp_ar

    # Storms cluster: a wet hour makes the next hour much more likely.
    p_wet = np.where(wet, 0.09, 0.018)
    rain_flag = np.zeros(n, dtype=bool)
    rain = np.zeros(n)
    for i in range(n):
        p = min(0.82, p_wet[i] * (3.4 if i and rain_flag[i - 1] else 1.0))
        rain_flag[i] = rng.random() < p
        if rain_flag[i]:
            rain[i] = rng.gamma(1.35, 1.45)
    # Typhoon Kristine-like passage: a single modeled system, not a historical replay.
    typhoon = (index >= TYPHOON_START) & (index <= TYPHOON_END)
    rain[typhoon] = rng.uniform(6, 18, size=int(typhoon.sum()))
    temp[typhoon] -= 2.4

    wind_ar = ar1(n, 0.90, 0.28, rng)
    wind = 2.3 + 0.55 * wet.astype(float) + 0.25 * np.sin(2 * np.pi * hour / 24) + wind_ar
    wind = np.clip(wind, 0.4, 7.5)
    wind[typhoon] = rng.uniform(11, 16, size=int(typhoon.sum()))

    # Amihan from the northeast in the dry season, Habagat from the southwest in the wet.
    base_dir = np.where(wet, 230.0, 40.0)
    wind_dir = (base_dir + rng.normal(0, 28, size=n)) % 360
    wind_dir[typhoon] = (rng.normal(280, 18, size=int(typhoon.sum()))) % 360

    humidity = 78 - 7 * heat - 8 * np.sin(2 * np.pi * (hour - 4) / 24) + rng.normal(0, 2.5, n)
    humidity[rain_flag] = np.maximum(humidity[rain_flag], 88)
    humidity[typhoon] = rng.uniform(90, 98, size=int(typhoon.sum()))
    humidity = np.clip(humidity, 42, 99)

    pressure = 1011.5 - 0.55 * (temp - 27) + rng.normal(0, 0.6, n)
    pressure[typhoon] = rng.uniform(988, 996, size=int(typhoon.sum()))

    return pd.DataFrame({
        "observed_at": index,
        "temp_c": temp,
        "humidity_pct": humidity,
        "wind_speed_ms": wind,
        "wind_dir_deg": wind_dir,
        "rainfall_mm": rain,
        "pressure_hpa": pressure,
        "typhoon": typhoon,
    })


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    clock = _clock(index)
    metro_weather = _weather(index, clock, rng)
    n = len(index)

    traffic_z = _z(clock["traffic"])
    photo_z = _z(clock["photo"])
    season = clock["season"]
    regional = ar1(n, 0.96, 0.045, rng)

    rain = metro_weather["rainfall_mm"].to_numpy()
    wind = metro_weather["wind_speed_ms"].to_numpy()
    typhoon = metro_weather["typhoon"].to_numpy()
    hour = clock["hour"]
    dow = clock["dow"]

    # Station microclimates ride on the metro weather, they do not invent a new climate.
    micro = {
        "roadside": dict(wind=0.85, temp=0.6, rain=0.95),
        "commercial roadside": dict(wind=0.90, temp=0.5, rain=0.95),
        "airport roadside": dict(wind=1.15, temp=0.4, rain=0.95),
        "mixed roadside": dict(wind=0.95, temp=0.3, rain=1.00),
        "urban center": dict(wind=0.90, temp=0.7, rain=1.00),
        "urban background": dict(wind=1.00, temp=0.1, rain=1.00),
        "valley residential": dict(wind=0.72, temp=0.2, rain=1.05),
        "coastal background": dict(wind=1.35, temp=-0.4, rain=1.00),
        "industrial port": dict(wind=1.25, temp=0.3, rain=0.95),
        "industrial": dict(wind=0.95, temp=0.5, rain=0.95),
        "commercial": dict(wind=1.05, temp=0.4, rain=1.00),
        "upland background": dict(wind=1.20, temp=-2.4, rain=1.15),
    }

    reading_frames = []
    weather_frames = []
    for station in STATIONS:
        sid = station["station_id"]
        kind = micro[station["station_type"]]
        local_noise = ar1(n, 0.88, 0.04, rng)
        wind_s = np.clip(wind * kind["wind"] + rng.normal(0, 0.15, n), 0.3, None)
        rain_s = rain * kind["rain"]
        temp_s = metro_weather["temp_c"].to_numpy() + kind["temp"] + rng.normal(0, 0.15, n)

        log_pm = (
            np.log(station["pm25"])
            + 0.30 * season
            + 0.38 * traffic_z
            - 0.20 * np.log1p(rain_s)
            - 0.10 * (np.minimum(wind_s, 6.0) - 2.4)
            + regional
            + local_noise
        )
        # Valley nights with light wind trap the evening peak a little longer.
        if station["station_type"] == "valley residential":
            calm_night = (hour >= 20) | (hour <= 6)
            calm_night &= wind_s < 1.6
            log_pm = log_pm + 0.18 * calm_night

        pm25 = np.exp(log_pm)
        coarse = station["coarse"] * np.exp(
            0.22 * traffic_z - 0.12 * np.log1p(rain_s) + 0.55 * local_noise + rng.normal(0, 0.05, n)
        )
        coarse = np.clip(coarse, 2, None)
        pm10 = pm25 + coarse

        log_no2 = (
            np.log(station["no2"])
            + 0.62 * traffic_z
            - 0.09 * (wind_s - 2.4)
            + 0.08 * season
            + 0.65 * regional
            + 0.8 * local_noise
        )
        no2 = np.exp(log_no2)

        # Ozone follows sunlight and is eaten where fresh NO2 is high.
        no2_z = _z(np.log(no2))
        log_o3 = (
            np.log(station["o3"])
            + 0.30 * photo_z
            - 0.28 * no2_z
            - 0.22 * (rain_s > 1).astype(float)
            + 0.28 * regional
            + 0.40 * local_noise
        )
        o3 = np.exp(log_o3)

        log_so2 = (
            np.log(station["so2"])
            + 0.18 * traffic_z
            + 0.04 * season
            + 0.4 * local_noise
        )
        so2 = np.exp(log_so2)

        log_co = (
            np.log(station["co"])
            + 0.58 * traffic_z
            - 0.07 * (wind_s - 2.4)
            + 0.7 * regional
            + 0.7 * local_noise
        )
        co = np.exp(log_co)

        # Typhoon: rain and wind already jumped; concentrations still need the washout,
        # because the log1p term alone does not drop a 12 mm hour far enough.
        # Rain is already inside the PM equation. This extra cut is the rest of a
        # typhoon: deep mixing and a wind that the ordinary dispersion term understates.
        pm25 = np.where(typhoon, pm25 * 0.85, pm25)
        pm10 = np.where(typhoon, pm10 * 0.88, pm10)
        no2 = np.where(typhoon, no2 * 0.70, no2)
        o3 = np.where(typhoon, o3 * 0.75, o3)
        so2 = np.where(typhoon, so2 * 0.72, so2)
        co = np.where(typhoon, co * 0.70, co)

        # Fireworks are fine-particle events. PM2.5 rises much more than coarse dust,
        # so the PM2.5/PM10 ratio climbs toward 0.9.
        for start, end, strength in FIREWORKS:
            mask = (index >= start) & (index <= end)
            bump = 1 + (station["spike"] - 1) * strength
            pm25 = np.where(mask, pm25 * bump, pm25)
            pm10 = np.where(mask, np.maximum(pm10 * (1 + 0.35 * (bump - 1)), pm25 * 1.08), pm10)

        # Construction dust at one industrial station, workdays, daytime only.
        # Coarse jumps. The combustion gases do not.
        if sid == CONSTRUCTION_STATION:
            work = (
                (index >= CONSTRUCTION_START)
                & (index <= CONSTRUCTION_END)
                & (dow < 5)
                & (hour >= 7)
                & (hour <= 17)
            )
            pm25 = np.where(work, pm25 * 1.18, pm25)
            pm10 = np.where(work, pm10 + 115, pm10)

        pm10 = np.maximum(pm10, pm25 * 1.04)
        pm25 = np.clip(pm25, 2, 480)
        pm10 = np.clip(pm10, 3, 600)
        no2 = np.clip(no2, 2, 320)
        o3 = np.clip(o3, 2, 260)
        so2 = np.clip(so2, 0.5, 120)
        co = np.clip(co, 0.05, 12)

        values = {"PM25": pm25, "PM10": pm10, "NO2": no2, "O3": o3, "SO2": so2, "CO": co}
        flags = {k: np.full(n, "ok", dtype=object) for k in values}

        fault = (index >= FAULT_START) & (index <= FAULT_END)
        if sid == FAULT_STATION and fault.any():
            first = int(np.argmax(fault))
            for key in ("PM25", "PM10"):
                frozen = values[key][first - 1]
                values[key] = np.where(fault, frozen, values[key])
                flags[key][fault] = "sensor_fault"

        # Telemetry gaps are independent of the frozen sensor. They are the ordinary
        # missingness a dashboard has to live with.
        for key in values:
            gap = rng.random(n) < GAP_RATE
            gap &= flags[key] == "ok"
            values[key] = np.where(gap, np.nan, values[key])
            flags[key][gap] = "telemetry_gap"

        for key, series in values.items():
            reading_frames.append(pd.DataFrame({
                "station_id": sid,
                "pollutant_id": key,
                "observed_at": index,
                "value": series,
                "quality_flag": flags[key],
            }))

        weather_frames.append(pd.DataFrame({
            "station_id": sid,
            "observed_at": index,
            "temp_c": np.round(temp_s, 1),
            "humidity_pct": np.round(np.clip(metro_weather["humidity_pct"].to_numpy() + rng.normal(0, 0.8, n), 40, 99), 1),
            "wind_speed_ms": np.round(wind_s, 1),
            "wind_dir_deg": np.round((metro_weather["wind_dir_deg"].to_numpy() + rng.normal(0, 8, n)) % 360, 0),
            "rainfall_mm": np.round(rain_s, 2),
            "pressure_hpa": np.round(metro_weather["pressure_hpa"].to_numpy() + rng.normal(0, 0.3, n), 1),
        }))

    readings = pd.concat(reading_frames, ignore_index=True)
    readings.insert(0, "reading_id", np.arange(1, len(readings) + 1))
    # One decimal is honest for these instruments, except CO which is carried in mg/m³.
    readings["value"] = readings["value"].astype(float)
    for pid, places in {"PM25": 1, "PM10": 1, "NO2": 1, "O3": 1, "SO2": 1, "CO": 2}.items():
        mask = readings["pollutant_id"] == pid
        readings.loc[mask, "value"] = readings.loc[mask, "value"].round(places)

    weather = pd.concat(weather_frames, ignore_index=True)
    weather.insert(0, "weather_id", np.arange(1, len(weather) + 1))

    alerts = _alerts(readings)
    stations = pd.DataFrame([{
        "station_id": s["station_id"],
        "station_name": s["name"],
        "city": s["city"],
        "district": s["district"],
        "station_type": s["station_type"],
        "latitude": s["lat"],
        "longitude": s["lon"],
        "elevation_m": s["elev"],
        "commissioned_on": s["commissioned"],
        "operator": "Synthetic Metro Manila Airshed",
    } for s in STATIONS])
    pollutants = pd.DataFrame(POLLUTANTS)

    return {
        "stations": stations,
        "pollutants": pollutants,
        "readings": readings,
        "weather_proxies": weather,
        "alerts": alerts,
    }


def _averaged(series: pd.Series, pollutant_id: str) -> pd.Series:
    """Match the averaging time the breakpoint was built for."""
    if pollutant_id in ("PM25", "PM10"):
        return series.rolling("24h", min_periods=18).mean()
    if pollutant_id in ("O3", "CO"):
        return series.rolling("8h", min_periods=6).mean()
    return series


def _alerts(readings: pd.DataFrame) -> pd.DataFrame:
    # Frozen PM must not open an alert, and telemetry gaps stay out of the mean.
    usable = readings[readings["quality_flag"] == "ok"]
    parts = []
    for (sid, pid), block in usable.groupby(["station_id", "pollutant_id"], sort=False):
        series = block.set_index("observed_at")["value"].sort_index()
        averaged = _averaged(series, pid).dropna()
        if averaged.empty:
            continue
        parts.append(pd.DataFrame({
            "station_id": sid,
            "pollutant_id": pid,
            "observed_at": averaged.index,
            "value": averaged.to_numpy(),
            "sub": subindex(averaged.to_numpy(dtype=float), pid),
        }))
    usable = pd.concat(parts, ignore_index=True)
    usable = usable.sort_values(["station_id", "observed_at", "sub"], ascending=[True, True, False])
    top = usable.groupby(["station_id", "observed_at"], as_index=False).first()
    top = top.rename(columns={"sub": "aqi", "pollutant_id": "driver"})
    top["level"] = [aqi_label(a) for a in top["aqi"]]

    rows = []
    alert_i = 1
    for sid, block in top.groupby("station_id", sort=False):
        block = block.sort_values("observed_at")
        hot = block["aqi"].to_numpy() >= 101
        times = block["observed_at"].to_numpy()
        aqis = block["aqi"].to_numpy()
        drivers = block["driver"].to_numpy()
        levels = block["level"].to_numpy()
        i = 0
        m = len(block)
        while i < m:
            if not hot[i]:
                i += 1
                continue
            j = i
            while j + 1 < m and hot[j + 1]:
                # Break the episode if the clock jumps by more than an hour (a gap).
                delta = (pd.Timestamp(times[j + 1]) - pd.Timestamp(times[j])).total_seconds()
                if delta > 3600 * 1.5:
                    break
                j += 1
            window = slice(i, j + 1)
            peak_at = i + int(np.nanargmax(aqis[window]))
            rows.append({
                "alert_id": f"ALT-{alert_i:04d}",
                "station_id": sid,
                "started_at": pd.Timestamp(times[i]),
                "cleared_at": pd.Timestamp(times[j]) + pd.Timedelta(hours=1),
                "hours": int(j - i + 1),
                "peak_aqi": int(round(aqis[peak_at])),
                "peak_at": pd.Timestamp(times[peak_at]),
                "driver_pollutant": drivers[peak_at],
                "peak_level": levels[peak_at],
            })
            alert_i += 1
            i = j + 1
    alerts = pd.DataFrame(rows)
    if len(alerts):
        alerts = alerts.sort_values(["started_at", "station_id"]).reset_index(drop=True)
        alerts["alert_id"] = [f"ALT-{i:04d}" for i in range(1, len(alerts) + 1)]
    return alerts


def sanity_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    readings = tables["readings"]
    weather = tables["weather_proxies"]
    alerts = tables["alerts"]
    ok = readings[readings["quality_flag"] == "ok"]
    wide = ok.pivot_table(index=["station_id", "observed_at"], columns="pollutant_id", values="value")
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("no negative concentrations", bool((ok["value"].dropna() >= 0).all()), f"min {ok['value'].min():.2f}")
    pair = wide.dropna(subset=["PM25", "PM10"])
    add("PM10 always above PM2.5", bool((pair["PM10"] >= pair["PM25"] - 0.05).all()), f"rows {len(pair):,}")
    r_pm = pair["PM25"].corr(pair["PM10"])
    add("PM2.5 correlates with PM10", r_pm > 0.75, f"r = {r_pm:.2f}")

    daily = (
        ok[ok["pollutant_id"] == "PM25"]
        .assign(day=lambda d: d["observed_at"].dt.floor("D"))
        .groupby(["station_id", "day"])["value"].mean()
        .rename("pm25")
        .reset_index()
    )
    rain_daily = (
        weather.assign(day=lambda d: d["observed_at"].dt.floor("D"))
        .groupby(["station_id", "day"])["rainfall_mm"].sum()
        .rename("rain")
        .reset_index()
    )
    joined = daily.merge(rain_daily, on=["station_id", "day"])
    # Pool stations so one odd site cannot hide the washout.
    r_rain = joined["pm25"].corr(joined["rain"])
    wet = joined.loc[joined["rain"] >= 5, "pm25"].mean()
    dry = joined.loc[joined["rain"] < 0.2, "pm25"].mean()
    add("rain washes PM2.5 down", (r_rain < -0.15) and (wet < dry * 0.85), f"daily r = {r_rain:.2f}; wet-day mean {wet:.1f} vs dry-day {dry:.1f}")

    wind_daily = (
        weather.assign(day=lambda d: d["observed_at"].dt.floor("D"))
        .groupby(["station_id", "day"])["wind_speed_ms"].mean()
        .rename("wind")
        .reset_index()
    )
    wj = daily.merge(wind_daily, on=["station_id", "day"])
    r_wind = wj["pm25"].corr(wj["wind"])
    add("wind disperses PM2.5", r_wind < -0.1, f"daily r = {r_wind:.2f}")

    no2 = ok[ok["pollutant_id"] == "NO2"].copy()
    no2["dow"] = no2["observed_at"].dt.dayofweek
    roadside = {"STA-EDSA", "STA-MKT", "STA-PSG", "STA-NAIA"}
    no2_r = no2[no2["station_id"].isin(roadside)]
    weekday = no2_r.loc[no2_r["dow"] < 5, "value"].mean()
    sunday = no2_r.loc[no2_r["dow"] == 6, "value"].mean()
    add("Sunday traffic is quieter", sunday < weekday * 0.88, f"roadside NO2 weekday {weekday:.1f} vs Sunday {sunday:.1f}")

    o3 = ok[ok["pollutant_id"] == "O3"].copy()
    o3["hour"] = o3["observed_at"].dt.hour
    h14 = o3.loc[o3["hour"] == 14, "value"].mean()
    h6 = o3.loc[o3["hour"] == 6, "value"].mean()
    add("ozone peaks in the afternoon", h14 > h6 * 1.4, f"14:00 mean {h14:.1f} vs 06:00 {h6:.1f}")

    day = wide.reset_index()
    day["hour"] = day["observed_at"].dt.hour
    daytime = day[(day["hour"].between(10, 16)) & day["station_id"].isin(roadside)]
    r_o3 = daytime["O3"].corr(daytime["NO2"])
    add("daytime ozone and NO2 oppose each other", r_o3 < -0.2, f"roadside 10:00–16:00 r = {r_o3:.2f}")

    annual = ok[ok["pollutant_id"] == "PM25"].groupby("station_id")["value"].mean()
    add("upland station is cleaner than EDSA", annual["STA-ANT"] < annual["STA-EDSA"] * 0.6, f"Antipolo {annual['STA-ANT']:.1f} vs EDSA {annual['STA-EDSA']:.1f}")
    add("annual means sit in a Manila-like band", bool(annual.between(8, 55).all()), annual.round(1).to_dict())

    # Edge 1 — fireworks
    nye = ok[(ok["station_id"] == "STA-EDSA") & (ok["pollutant_id"] == "PM25")]
    spike = nye[nye["observed_at"].between("2024-01-01", "2024-01-01 04:00")]["value"].mean()
    baseline = nye[(nye["observed_at"] >= "2024-01-08") & (nye["observed_at"] < "2024-01-15")]["value"].median()
    add("edge: New Year fireworks at EDSA", spike > baseline * 3, f"Jan 1 00:00–04:00 mean {spike:.0f} vs a quiet week median {baseline:.0f}")

    # Edge 2 — frozen sensor
    fault = readings[(readings["station_id"] == FAULT_STATION) & (readings["pollutant_id"] == "PM25") & (readings["observed_at"].between(FAULT_START, FAULT_END))]
    add("edge: EDSA PM2.5 sensor freezes in August", bool((fault["quality_flag"] == "sensor_fault").all() and fault["value"].nunique() == 1), f"unique values {fault['value'].nunique()} over {len(fault)} hours")

    # Edge 3 — construction dust changes the ratio, not the gases
    val_ok = ok[ok["station_id"] == CONSTRUCTION_STATION].copy()
    val_wide = val_ok.pivot_table(index="observed_at", columns="pollutant_id", values="value")
    work = (val_wide.index >= CONSTRUCTION_START) & (val_wide.index <= CONSTRUCTION_END)
    work &= val_wide.index.dayofweek < 5
    work &= val_wide.index.hour.to_series().between(7, 17).to_numpy()
    ratio_work = (val_wide.loc[work, "PM10"] / val_wide.loc[work, "PM25"]).median()
    ordinary = ~((val_wide.index >= "2024-03-01") & (val_wide.index <= "2024-03-20"))
    ratio_base = (val_wide.loc[ordinary, "PM10"] / val_wide.loc[ordinary, "PM25"]).median()
    add("edge: Valenzuela construction is coarse dust", ratio_work > ratio_base * 1.6, f"PM10/PM2.5 median {ratio_work:.2f} on site vs {ratio_base:.2f} otherwise")

    # Edge 4 — typhoon
    typhoon_mean = nye[nye["observed_at"].between(TYPHOON_START, TYPHOON_END)]["value"].mean()
    around = nye[nye["observed_at"].between("2024-09-01", "2024-09-10")]["value"].mean()
    add("edge: typhoon week collapses EDSA PM2.5", typhoon_mean < around * 0.55, f"typhoon mean {typhoon_mean:.1f} vs early September {around:.1f}")

    add("alerts exist and are episode-length", (len(alerts) > 20) and (alerts["hours"].median() >= 2), f"{len(alerts)} episodes, median {alerts['hours'].median():.0f} h, peak {alerts['peak_aqi'].max()}")
    pm_share = (alerts["driver_pollutant"] == "PM25").mean()
    add("PM2.5 drives most alerts", pm_share > 0.45, f"PM2.5 share of episodes {pm_share:.0%}")

    return pd.DataFrame(checks)
