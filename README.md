# SD-147 · Urban air quality dashboard

A synthetic year of Metro Manila air, hourly, for twelve stations. Built as an Eskwelabs EIF **Synthetic Dataset** project: a notebook, the CSVs it writes, and a written record of every assumption.

The pictures are the point. Roadside EDSA and upland Antipolo share a calendar and do not share a color. Rush hour, Sunday, rain, a firework night, a dust job, a frozen sensor, and a typhoon are all visible without opening the tables.

This is not DENR-EMB data and it is not a regulatory AQI. Seed `147` reproduces the files exactly.

## Run it

From the repo root, with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook SD-147_urban_air_quality.ipynb
```

Run All. The notebook writes `data/*.csv` and stops if a sanity check fails. The full specification of the generator is in [assumptions.md](assumptions.md). The code is in `src/aq_model.py`.

## Tables

| File | What it is |
| --- | --- |
| `data/stations.csv` | 12 sites, with type, coordinates, elevation |
| `data/pollutants.csv` | PM2.5, PM10, NO2, ozone, SO2, CO, and their units |
| `data/readings.csv` | one row per station, pollutant, and hour, with a quality flag |
| `data/weather_proxies.csv` | temperature, humidity, wind, rain, pressure at the same hours |
| `data/alerts.csv` | episodes where the dashboard index reached 101 |

Quality flags are `ok`, `telemetry_gap`, and `sensor_fault`.

## What you can analyse with it

- Which stations, and which months, sit above the 24-hour PM2.5 breakpoints, and how much of that is dry-season traffic rather than a bad instrument.
- The weekday rush-hour shape of PM2.5 and NO2 against the afternoon shape of ozone, and how far Sunday sits below a weekday at a roadside site.
- Whether rain and wind actually clear the basin, and by how much once you look at daily totals rather than single hours.
- How often alerts fire, how long they last, and when the driver is particles rather than NO2.
- Three data-quality traps a dashboard has to survive: firework nights that look like combustion, a construction site that inflates PM10 only, and a week in August at EDSA where the PM channel is a flat line.

## What will break it

Changing the seed changes every row, including the planted edge cases, which are deterministic given the seed but sit on top of the random weather. Dropping `quality_flag` and averaging the August week at EDSA will report a clean week that never happened. Treating the index as a legal AQI will overstate it: particles are judged on a trailing 24-hour mean, and the breakpoint table is the US EPA table, not DENR's.
