# Assumptions — SD-147 Urban air quality dashboard

Synthetic hourly air quality for one Metro Manila airshed in 2024. Nothing here is a DENR-EMB observation, a regulatory AQI, or a forecast. The seed is **147**. A fresh run of the notebook reproduces every row.

The audience is an urban-services analyst who needs a dashboard: where the air is bad, when it is bad, and which hours not to trust.

## What is generated

| Table | Grain | Rows, roughly |
| --- | --- | --- |
| `stations` | one monitoring site | 12 |
| `pollutants` | one measured species | 6 |
| `readings` | station × pollutant × hour | ~630,000 |
| `weather_proxies` | station × hour | ~105,000 |
| `alerts` | one episode above the index threshold | hundreds |

Coordinates are approximate landmark locations so a map has a real shape. The operator field is `Synthetic Metro Manila Airshed` on purpose.

## The process

Every concentration is lognormal. The log-mean is a station baseline plus a few shared drivers, then two autoregressive shocks so neighbouring hours look like weather rather than white noise.

Shared drivers, built once for the year:

- **Traffic.** Sum of two Gaussians, centered at 08:00 (width 1.45 h) and 18:30 (width 1.9 h), plus a night floor of 0.16. Saturday multiplies by 0.84, Sunday by 0.68, a public holiday by 0.55. Holy Week (24–31 March) is held to at most 0.62.
- **Sunlight.** A Gaussian centered at 14:00, width 2.15 h. Ozone uses this. Particles use traffic instead.
- **Season.** A cosine peaking on day-of-year 40 (early February), the dry Amihan months. A second cosine, peaking on day-of-year 120, is only for temperature.
- **Regional shock.** AR(1), phi 0.96, innovation 0.045. This is a bad-air day that all twelve stations feel together.
- **Local shock.** AR(1), phi 0.88, innovation 0.04, drawn separately at each station.

Rain is a clustered process. In the wet season (day-of-year 152–334) an hour is wet with probability 0.09, otherwise 0.018. A wet hour multiplies the next hour's probability by 3.4, capped at 0.82. A wet hour's amount is Gamma(shape 1.35, scale 1.45) millimetres. Wind is an AR(1) around 2.3 m/s, higher in the wet season, and is capped at 6 m/s inside the dispersion term so a typhoon does not multiply concentrations down to the instrument floor.

Station microclimates scale that metro weather. The coast and the port are windier. Antipolo is 2.4°C cooler and a little wetter. Marikina is calmer, and on a calm night (20:00–06:00, wind under 1.6 m/s) its PM2.5 log-mean is lifted by 0.18, a stand-in for drainage flow into the valley.

### PM2.5

```
log = log(baseline)
    + 0.30 * season
    + 0.38 * traffic_z
    - 0.20 * log1p(rain)
    - 0.10 * (min(wind, 6) - 2.4)
    + regional + local
```

`traffic_z` is the traffic wave after it has been standardised. Baselines, in µg/m³, are the geometric level of a calm hour before the diurnal cycle: EDSA 33, Valenzuela 35, Port 31, Marikina 30, Manila 29, Pasig 28, Makati CBD 26, NAIA 25, Quezon City 22, BGC 19, Las Piñas 16, Antipolo 11.

### PM10

PM10 is PM2.5 plus a coarse-dust term that has its own baseline (EDSA 24 µg/m³, Valenzuela 30, Antipolo 7, and so on). Coarse dust follows traffic and is washed by rain, with a wider local shock than the fine fraction. After every edit, PM10 is forced to at least 1.04 × PM2.5. The two series move together, and the ratio is the thing that breaks during construction.

### Gases

- **NO2** follows traffic with coefficient 0.62 and disperses with wind. Roadside baselines are 40–50 µg/m³. Antipolo is 10.
- **CO** (mg/m³) is the same idea with coefficient 0.58. It is the particle series' cousin, not an independent story.
- **Ozone** follows the 14:00 sunlight curve (coefficient 0.30) and is reduced where NO2 is high (coefficient −0.28 on the log-NO2 z-score). An hour with more than 1 mm of rain is cut further. Background and upland stations have higher baselines (Antipolo 54, Las Piñas 48, Quezon City 44) because there is less fresh NO to titrate the ozone. Roadside baselines sit near 30–34.
- **SO2** is mostly the station baseline. The port is 18 µg/m³, Valenzuela 15, everyone else 3–8. Traffic only nudges it.

Values are then clipped to a physical range: PM2.5 2–480, PM10 3–600, NO2 2–320, ozone 2–260, SO2 0.5–120, CO 0.05–12. Reported precision is one decimal, two for CO.

## Edge cases, planted on purpose

These are not draws from the tails. The code writes them after the base process.

1. **Fireworks, fine mode.** 1 January 00:00–05:00 at full strength, 10 February 19:00–23:00 at 0.42 of that strength, 31 December 18:00–23:00 at 0.85. Each station has its own spike factor (EDSA 13, Antipolo 3.2). PM2.5 is multiplied through. PM10 is lifted less, and then held to at least 1.08 × the new PM2.5, so the fine fraction dominates the way firework smoke actually does.
2. **Construction dust at Valenzuela.** Weekdays 4–15 March, 07:00–17:00. PM2.5 × 1.18. PM10 gains 115 µg/m³ of coarse material. NO2, ozone, SO2, and CO are untouched.
3. **Frozen PM sensor at EDSA.** 12 August 00:00 through 18 August 23:00. PM2.5 and PM10 are held at the last good hour and flagged `sensor_fault`. The gas channels keep their ordinary values and stay `ok`. Alerts ignore the frozen channels.
4. **Typhoon passage.** 15 September 06:00 through 17 September 18:00, every station. Rain is redrawn between 6 and 18 mm each hour, wind between 11 and 16 m/s, temperature down 2.4°C, humidity 90–98%, pressure 988–996 hPa. On top of the rain and wind already in the equation, concentrations are multiplied by 0.85 (PM2.5), 0.88 (PM10), 0.70 (NO2), 0.75 (ozone), 0.72 (SO2), 0.70 (CO). It is one synthetic system, not a reconstruction of a named storm.

Separately, 0.4% of hours that are still flagged `ok` are turned into `telemetry_gap` with a missing value. Gaps are independent across pollutants.

## The index and the alerts

Breakpoints follow the US EPA tables, converted at 25°C where the standard is published in ppb or ppm.

- PM2.5 and PM10 use the 24-hour table, on a trailing 24-hour mean (at least 18 hours present).
- Ozone and CO use the 8-hour table, on a trailing 8-hour mean (at least 6 hours).
- NO2 and SO2 use the 1-hour table, on the hour itself.

The calendar colours the **daily mean** of PM2.5 with the 24-hour PM2.5 breakpoints: Good through 12.0, Moderate through 35.4, Unhealthy for sensitive groups through 55.4, then Unhealthy, Very unhealthy, Hazardous. Grey on that calendar is a day whose PM channel was frozen for at least 12 hours. It is not a clean day.

An alert opens when the index reaches 101 and closes when it drops below, or when the clock jumps by more than an hour and a half. The driver is the pollutant with the highest sub-index at the peak hour. `sensor_fault` and `telemetry_gap` hours do not enter the average and cannot open an episode.

This over-states a legal AQI whenever the trailing window is only partly full, and it is not the DENR breakpoint table. It is a consistent dashboard index.

## What was left out

No chemistry beyond the NO2 titration of ozone. No street-canyon model, no sea-breeze clock, no transboundary smoke except the three firework nights. Holiday dates are the 2024 national set used to quiet traffic, not a claim about work-from-home or mall hours. Instrument drift, other than the one frozen week and the random gaps, is not modeled.

## Sanity bar

The notebook refuses to finish if any of these fail:

- no negative concentrations
- PM10 stays at or above PM2.5
- PM2.5 and PM10 correlate above 0.75
- wet days are cleaner than dry days, and the daily PM–rain correlation is negative
- daily PM falls as wind rises
- Sunday roadside NO2 is at least 12% under the weekday mean
- ozone at 14:00 is at least 1.4× ozone at 06:00
- daytime roadside ozone and NO2 correlate below −0.2
- Antipolo's annual PM2.5 is under 60% of EDSA's
- every station's annual PM2.5 lands between 8 and 55 µg/m³
- the New Year window at EDSA is more than 3× a quiet January week
- the August fault is a single repeated value, flagged on every hour
- the Valenzuela work-hour PM10/PM2.5 ratio is more than 1.6× the station's ordinary ratio
- the typhoon window at EDSA is under 55% of early September
- alerts exist, last more than a couple of hours at the median, and PM2.5 drives most of them
