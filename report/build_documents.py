"""Build the SD-147 PDF report and the 16:9 slide deck.

Run from the repo root:

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python report/build_documents.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from string import Template

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aq_plots import (  # noqa: E402
    CLAY,
    INK,
    MUTED,
    PAPER,
    PM_CMAP,
    TEAL,
    apply_style,
    calendars,
    diurnal,
    episodes,
    season_grid,
    station_map,
    washout,
)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"

NAMES = {
    "PM25": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "O3": "ozone",
    "SO2": "SO2",
    "CO": "CO",
}


def load() -> dict[str, pd.DataFrame]:
    data = ROOT / "data"
    return {
        "stations": pd.read_csv(data / "stations.csv"),
        "pollutants": pd.read_csv(data / "pollutants.csv"),
        "readings": pd.read_csv(data / "readings.csv", parse_dates=["observed_at"]),
        "weather_proxies": pd.read_csv(data / "weather_proxies.csv", parse_dates=["observed_at"]),
        "alerts": pd.read_csv(data / "alerts.csv", parse_dates=["started_at", "cleared_at", "peak_at"]),
    }


def _daily(readings: pd.DataFrame, station_id: str) -> pd.DataFrame:
    sub = readings[(readings["station_id"] == station_id) & (readings["pollutant_id"] == "PM25")].copy()
    sub["day"] = sub["observed_at"].dt.floor("D")
    rows = []
    for day, block in sub.groupby("day"):
        fault_hours = int((block["quality_flag"] == "sensor_fault").sum())
        ok = block.loc[block["quality_flag"] == "ok", "value"]
        rows.append({"day": day, "mean": ok.mean() if len(ok) else np.nan, "fault": fault_hours >= 12})
    return pd.DataFrame(rows)


def facts(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    readings = tables["readings"]
    weather = tables["weather_proxies"]
    alerts = tables["alerts"]
    stations = tables["stations"].set_index("station_id")
    ok = readings[readings["quality_flag"] == "ok"]

    annual = (
        ok[ok["pollutant_id"] == "PM25"].groupby("station_id")["value"].mean().rename("pm25")
    )
    episode_n = alerts.groupby("station_id").size().rename("episodes")
    board = stations.join(annual).join(episode_n)
    board["episodes"] = board["episodes"].fillna(0)
    board = board.sort_values("pm25", ascending=False)

    wide = ok.pivot_table(index=["station_id", "observed_at"], columns="pollutant_id", values="value")
    pair = wide.dropna(subset=["PM25", "PM10"])
    r_pm = pair["PM25"].corr(pair["PM10"])

    daily = (
        ok[ok["pollutant_id"] == "PM25"]
        .assign(day=lambda d: d["observed_at"].dt.floor("D"))
        .groupby(["station_id", "day"])["value"].mean()
        .rename("pm25")
        .reset_index()
    )
    day_weather = weather.assign(day=lambda d: d["observed_at"].dt.floor("D"))
    rain = day_weather.groupby(["station_id", "day"])["rainfall_mm"].sum().rename("rain").reset_index()
    wind = day_weather.groupby(["station_id", "day"])["wind_speed_ms"].mean().rename("wind").reset_index()
    joined = daily.merge(rain, on=["station_id", "day"]).merge(wind, on=["station_id", "day"])
    wet = joined.loc[joined["rain"] >= 5, "pm25"].mean()
    dry = joined.loc[joined["rain"] < 0.2, "pm25"].mean()
    heavy = joined.loc[joined["rain"] >= 40, "pm25"].mean()

    roadside = {"STA-EDSA", "STA-MKT", "STA-PSG", "STA-NAIA"}
    no2 = ok[ok["pollutant_id"] == "NO2"].copy()
    no2["dow"] = no2["observed_at"].dt.day_of_week
    no2_r = no2[no2["station_id"].isin(roadside)]
    weekday = no2_r.loc[no2_r["dow"] < 5, "value"].mean()
    sunday = no2_r.loc[no2_r["dow"] == 6, "value"].mean()

    o3 = ok[ok["pollutant_id"] == "O3"].copy()
    o3["hour"] = o3["observed_at"].dt.hour
    h14 = o3.loc[o3["hour"] == 14, "value"].mean()
    h6 = o3.loc[o3["hour"] == 6, "value"].mean()
    day = wide.reset_index()
    day["hour"] = day["observed_at"].dt.hour
    daytime = day[(day["hour"].between(10, 16)) & day["station_id"].isin(roadside)]

    edsa = _daily(readings, "STA-EDSA").dropna(subset=["mean"])
    ant = _daily(readings, "STA-ANT").dropna(subset=["mean"])
    fault_days = int(_daily(readings, "STA-EDSA")["fault"].sum())

    nye = ok[(ok["station_id"] == "STA-EDSA") & (ok["pollutant_id"] == "PM25")]
    spike = nye[nye["observed_at"].between("2024-01-01", "2024-01-01 04:00")]["value"].mean()
    quiet = nye[(nye["observed_at"] >= "2024-01-08") & (nye["observed_at"] < "2024-01-15")]["value"].median()
    typhoon = nye[nye["observed_at"].between("2024-09-15 06:00", "2024-09-17 18:00")]["value"].mean()
    early = nye[nye["observed_at"].between("2024-09-01", "2024-09-10")]["value"].mean()

    val = ok[ok["station_id"] == "STA-VAL"].pivot_table(
        index="observed_at", columns="pollutant_id", values="value"
    )
    work = (val.index >= "2024-03-04") & (val.index <= "2024-03-15 23:00")
    work &= val.index.day_of_week < 5
    work &= val.index.hour.to_series().between(7, 17).to_numpy()
    ordinary = ~((val.index >= "2024-03-01") & (val.index <= "2024-03-20"))
    ratio_work = (val.loc[work, "PM10"] / val.loc[work, "PM25"]).median()
    ratio_base = (val.loc[ordinary, "PM10"] / val.loc[ordinary, "PM25"]).median()

    freeze = readings[
        (readings["station_id"] == "STA-EDSA")
        & (readings["pollutant_id"] == "PM25")
        & (readings["observed_at"].between("2024-08-12", "2024-08-18 23:00"))
    ]
    worst = alerts.loc[alerts["peak_aqi"].idxmax()]
    longest = alerts.loc[alerts["hours"].idxmax()]
    drivers = alerts["driver_pollutant"].value_counts()
    levels = alerts["peak_level"].value_counts()
    gaps = int((readings["quality_flag"] == "telemetry_gap").sum())
    top = board["episodes"].sort_values(ascending=False).head(3)

    driver_bits = []
    for key, count in drivers.items():
        driver_bits.append(f"{NAMES.get(key, key)} {int(count):,}")
    level_bits = [
        f"{int(count):,} episodes peak at {key}" if i == 0 else f"{int(count):,} peak at {key}"
        for i, (key, count) in enumerate(levels.items())
    ]
    level_sentence = level_bits[0] if len(level_bits) == 1 else ", ".join(level_bits[:-1]) + f", and {level_bits[-1]}"
    level_sentence = level_sentence[0].upper() + level_sentence[1:] + "."

    rows = []
    for sid, row in board.iterrows():
        width = float(row["pm25"]) / 42 * 100
        rows.append(
            "<tr>"
            f"<td>{row['station_name']}</td>"
            f"<td class='place'>{row['city']}</td>"
            f"<td class='place'>{row['station_type']}</td>"
            f"<td class='num'>{row['pm25']:.0f}</td>"
            f"<td class='meter'><span style='width:{width:.1f}%'></span></td>"
            f"<td class='num'>{int(row['episodes'])}</td>"
            "</tr>"
        )

    ozone_note = (
        "Ozone never opens an episode. The index uses an eight-hour mean, so a single afternoon spike does not qualify."
        if "O3" not in drivers.index
        else f"Ozone opens {int(drivers['O3']):,} episodes."
    )

    return {
        "n_readings": f"{len(readings):,}",
        "n_weather": f"{len(weather):,}",
        "n_alerts": f"{len(alerts):,}",
        "n_gaps": f"{gaps:,}",
        "gap_pct": f"{100 * gaps / len(readings):.1f}",
        "stations_html": "\n".join(rows),
        "edsa_pm": f"{annual['STA-EDSA']:.0f}",
        "ant_pm": f"{annual['STA-ANT']:.0f}",
        "val_pm": f"{annual['STA-VAL']:.0f}",
        "edsa_pm1": f"{annual['STA-EDSA']:.1f}",
        "ant_pm1": f"{annual['STA-ANT']:.1f}",
        "edsa_med": f"{edsa['mean'].median():.0f}",
        "edsa_above": f"{100 * (edsa['mean'] > 35.4).mean():.0f}",
        "edsa_high": f"{100 * (edsa['mean'] > 55.4).mean():.0f}",
        "edsa_max": f"{edsa['mean'].max():.0f}",
        "ant_med": f"{ant['mean'].median():.0f}",
        "ant_max": f"{ant['mean'].max():.0f}",
        "fault_days": str(fault_days),
        "r_pm": f"{r_pm:.2f}",
        "r_rain": f"{joined['pm25'].corr(joined['rain']):.2f}",
        "r_wind": f"{joined['pm25'].corr(joined['wind']):.2f}",
        "wet": f"{wet:.0f}",
        "dry": f"{dry:.0f}",
        "heavy": f"{heavy:.0f}",
        "weekday_no2": f"{weekday:.0f}",
        "sunday_no2": f"{sunday:.0f}",
        "sunday_cut": f"{100 * (1 - sunday / weekday):.0f}",
        "o3_14": f"{h14:.0f}",
        "o3_6": f"{h6:.0f}",
        "r_o3": f"{daytime['O3'].corr(daytime['NO2']):.2f}",
        "spike": f"{spike:.0f}",
        "quiet": f"{quiet:.0f}",
        "typhoon": f"{typhoon:.0f}",
        "early": f"{early:.0f}",
        "ratio_work": f"{ratio_work:.1f}",
        "ratio_base": f"{ratio_base:.1f}",
        "freeze_value": f"{freeze['value'].iloc[0]:.0f}",
        "freeze_hours": str(len(freeze)),
        "median_h": f"{alerts['hours'].median():.0f}",
        "max_h": f"{alerts['hours'].max():.0f}",
        "longest_days": f"{longest['hours'] / 24:.0f}",
        "longest_where": stations.loc[longest["station_id"], "station_name"],
        "peak": f"{worst['peak_aqi']:.0f}",
        "peak_when": pd.Timestamp(worst["peak_at"]).strftime("%-d %B"),
        "peak_where": stations.loc[worst["station_id"], "station_name"],
        "pm_share": f"{100 * (alerts['driver_pollutant'] == 'PM25').mean():.0f}",
        "drivers": ", ".join(driver_bits),
        "levels": level_sentence,
        "ozone_note": ozone_note,
        "top_alerts": ", ".join(
            f"{stations.loc[sid, 'station_name']} ({int(n)})" for sid, n in top.items()
        ),
        "ant_episodes": str(int(board.loc["STA-ANT", "episodes"])),
        "ant_alert": (
            "Antipolo and Las Piñas open none."
            if int(board.loc["STA-ANT", "episodes"]) == 0
            else f"Antipolo opens {int(board.loc['STA-ANT', 'episodes'])}."
        ),
        "edsa_episodes": str(int(episode_n["STA-EDSA"])),
    }


def save_figures(tables: dict[str, pd.DataFrame]) -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    made = {
        "calendars": calendars(tables["readings"]),
        "map": station_map(tables["stations"], tables["readings"], tables["alerts"]),
        "diurnal": diurnal(tables["readings"]),
        "episodes": episodes(tables["readings"], tables["weather_proxies"]),
        "season": season_grid(tables["readings"], tables["stations"]),
        "washout": washout(tables["readings"], tables["weather_proxies"]),
    }
    for name, fig in made.items():
        fig.savefig(FIG / f"{name}.png", dpi=180)
        plt.close(fig)
    _drivers(tables["alerts"])


def _drivers(alerts: pd.DataFrame) -> None:
    apply_style()
    order = ["PM25", "NO2", "PM10", "SO2", "CO", "O3"]
    counts = alerts["driver_pollutant"].value_counts()
    keys = [k for k in order if k in counts.index] + [k for k in counts.index if k not in order]
    values = [int(counts[k]) for k in keys]
    labels = [NAMES.get(k, k) for k in keys]
    color = {"PM2.5": "#B84332", "NO2": TEAL, "PM10": "#E08A3C", "SO2": MUTED, "CO": MUTED, "ozone": "#2F9E78"}
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=[color.get(label, MUTED) for label in labels], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for yi, value in zip(y, values):
        ax.text(value + max(values) * 0.02, yi, f"{value:,}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    ax.tick_params(length=0)
    ax.set_xlabel("Episodes")
    ax.set_title("What opened the episode", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "drivers.png", dpi=180)
    plt.close(fig)


REPORT_CSS = r"""
@page { size: A4; margin: 0; }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
html, body { margin: 0; padding: 0; background: #F6F1E8; color: #1C1917; }
body { counter-reset: pg; font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif; }
.sheet {
  width: 210mm; min-height: 297mm; padding: 14mm 16mm 13mm;
  page-break-after: always; break-after: page; position: relative;
}
.sheet:last-child { page-break-after: auto; break-after: auto; }
.sheet:not(.cover) { counter-increment: pg; }
.run {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase;
  color: #78716C; margin-bottom: 8mm;
}
.folio::before { content: counter(pg, decimal-leading-zero); }
h1, h2, .stat b, .cover-year { font-family: "Iowan Old Style", Palatino, Georgia, serif; font-weight: normal; }
h2 { font-size: 28px; line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 4mm; }
p { font-size: 12px; line-height: 1.48; margin: 0 0 3.2mm; }
.lede { font-size: 14.5px; line-height: 1.4; max-width: 42em; }
.quiet { color: #44403C; }
.small { font-size: 10.5px; line-height: 1.45; color: #44403C; }
.kicker {
  font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
  color: #0F6E6B; font-weight: 600; margin: 0 0 6mm;
}
.rule { height: 1px; background: #E4DDD2; border: 0; margin: 6mm 0; }
.cover { display: flex; flex-direction: column; }
.cover h1 { font-size: 64px; line-height: 0.92; letter-spacing: -0.03em; margin: 8mm 0 6mm; }
.cover .deck { font-size: 18px; line-height: 1.35; max-width: 34em; color: #292524; }
.stats { display: flex; gap: 10mm; margin-top: auto; padding-top: 10mm; }
.stat b { display: block; font-size: 34px; line-height: 1; }
.stat span { display: block; margin-top: 1.5mm; font-size: 11px; letter-spacing: 0.04em; color: #78716C; }
.swatches { display: flex; gap: 2.5mm; margin-top: 8mm; }
.swatches span {
  font-size: 9px; letter-spacing: 0.04em; color: #1C1917;
  padding: 1.4mm 2.4mm 1.2mm; border-radius: 99px;
}
.disclaimer { margin-top: 6mm; font-size: 11px; color: #78716C; }
.columns { display: flex; gap: 8mm; }
.columns .main { flex: 1.4; }
.columns .side { flex: 0.85; }
.card, .note {
  background: #EFE8DC; border-radius: 8px; padding: 4mm 4.5mm 3.5mm; margin: 0 0 3mm;
}
.card h3 { margin: 0 0 1.2mm; font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; color: #0F6E6B; }
.card p, .note p { font-size: 11.5px; margin: 0; }
pre {
  font-family: "Iowan Old Style", Palatino, Georgia, serif;
  font-size: 13px; line-height: 1.45; background: #EFE8DC;
  border-radius: 8px; padding: 5mm 6mm; margin: 2mm 0 4mm; white-space: pre;
}
table { width: 100%; border-collapse: collapse; margin-top: 2mm; }
th {
  text-align: left; font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
  color: #78716C; font-weight: 600; padding: 0 2mm 2mm 0; border-bottom: 1px solid #E4DDD2;
}
td { font-size: 11.5px; padding: 1.7mm 2mm 1.7mm 0; border-bottom: 1px solid #E4DDD2; vertical-align: middle; }
td.place { color: #44403C; }
td.num { text-align: right; font-variant-numeric: tabular-nums; width: 12mm; }
td.meter { width: 28mm; }
td.meter span { display: block; height: 6px; border-radius: 99px; background: #C2412D; }
img.full { width: 100%; height: auto; display: block; margin: 1mm 0 3mm; }
img.drivers { width: 118mm; height: auto; }
.pull { border-left: 2.5px solid #C2412D; padding-left: 4mm; margin: 4mm 0; }
.pull p { font-family: "Iowan Old Style", Palatino, Georgia, serif; font-size: 16px; line-height: 1.3; }
ol.edges { margin: 2mm 0 0; padding: 0; list-style: none; }
ol.edges li { display: flex; gap: 4mm; margin: 0 0 3mm; }
ol.edges b {
  font-family: "Iowan Old Style", Palatino, Georgia, serif;
  font-size: 18px; font-weight: normal; color: #C2412D; width: 8mm; flex: none;
}
ol.edges p { margin: 0; }
.end { margin-top: 6mm; }
"""

SLIDE_CSS = r"""
@page { size: 13.333in 7.5in; margin: 0; }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
html, body { margin: 0; padding: 0; background: #F6F1E8; color: #1C1917; }
body { font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif; }
.slide {
  width: 13.333in; height: 7.5in; padding: 0.42in 0.58in 0.36in;
  page-break-after: always; break-after: page; overflow: hidden;
  display: flex; flex-direction: column; position: relative;
}
.slide:last-child { page-break-after: auto; break-after: auto; }
h1, h2, .giant, .num b, .pill b, .force h3 {
  font-family: "Iowan Old Style", Palatino, Georgia, serif; font-weight: normal;
}
.kicker {
  font-size: 13px; letter-spacing: 0.18em; text-transform: uppercase;
  color: #0F6E6B; font-weight: 600; margin: 0;
}
h1 { font-size: 68px; line-height: 0.94; letter-spacing: -0.03em; margin: 0.18in 0 0; }
h2 { font-size: 34px; line-height: 1.02; letter-spacing: -0.02em; margin: 0.08in 0 0; }
.deck { font-size: 22px; line-height: 1.35; max-width: 10.2in; margin: 0.18in 0 0; color: #292524; }
.sub { margin-top: auto; font-size: 15px; letter-spacing: 0.08em; text-transform: uppercase; color: #78716C; }
img.fig { width: 100%; max-height: 5.35in; object-fit: contain; object-position: left center; margin-top: 0.08in; }
.caption { margin-top: auto; font-size: 18px; line-height: 1.35; max-width: 12in; padding-top: 0.08in; }
.pills { display: flex; gap: 0.16in; margin-top: 0.42in; }
.pill { flex: 1; background: #EFE8DC; border-radius: 12px; padding: 0.28in 0.2in 0.24in; }
.pill em { display: block; font-style: normal; font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: #78716C; }
.pill b { display: block; margin-top: 0.12in; font-size: 34px; line-height: 1; }
.pill span { display: block; margin-top: 0.1in; font-size: 15px; line-height: 1.3; color: #44403C; }
.forces { display: grid; grid-template-columns: 1fr 1fr; gap: 0.16in; margin-top: 0.28in; }
.force { background: #EFE8DC; border-radius: 12px; padding: 0.22in 0.26in 0.2in; }
.force h3 { margin: 0; font-size: 28px; }
.force p { margin: 0.08in 0 0; font-size: 16px; line-height: 1.35; color: #44403C; }
.aside { margin-top: auto; font-size: 18px; line-height: 1.35; }
.nums { display: grid; grid-template-columns: 1fr 1fr; gap: 0.28in 0.6in; margin-top: 0.34in; }
.num b { display: block; font-size: 64px; line-height: 0.9; }
.num span { display: block; margin-top: 0.08in; font-size: 18px; line-height: 1.3; max-width: 5.2in; color: #292524; }
.limits { margin-top: 0.28in; }
.limits div { display: flex; gap: 0.24in; align-items: baseline; margin: 0 0 0.2in; }
.limits b { font-family: "Iowan Old Style", Palatino, Georgia, serif; font-weight: normal; font-size: 28px; color: #C2412D; width: 0.6in; }
.limits p { margin: 0; font-size: 22px; line-height: 1.3; }
.split { display: flex; gap: 0.32in; margin-top: 0.12in; min-height: 0; flex: 1; }
.split img { width: 8.15in; height: 5.7in; object-fit: contain; object-position: left top; }
.split ol { list-style: none; margin: 0.15in 0 0; padding: 0; flex: 1; }
.split li { margin: 0 0 0.22in; }
.split strong { display: block; font-size: 18px; font-weight: 600; }
.split span { display: block; margin-top: 0.03in; font-size: 14.5px; line-height: 1.3; color: #44403C; }
.asks { margin: 0.28in 0 0; padding: 0; list-style: none; }
.asks li { font-size: 22px; line-height: 1.25; margin: 0 0 0.16in; padding-left: 0.32in; position: relative; }
.asks li::before { content: ""; position: absolute; left: 0; top: 0.16in; width: 0.14in; height: 0.14in; border-radius: 99px; background: #0F6E6B; }
"""


def sheet(body: str) -> str:
    return (
        '<section class="sheet"><header class="run">'
        "<span>SD-147 · Urban air quality</span><span class='folio'></span>"
        f"</header>{body}</section>"
    )


def report_html(f: dict[str, str]) -> str:
    pages = []
    pages.append(f"""
    <section class="sheet cover">
      <p class="kicker">Eskwelabs EIF · Synthetic dataset · SD-147</p>
      <h1>A year of<br>Metro Manila air.</h1>
      <p class="deck">Twelve stations, every hour of 2024, generated so a bad day, a clean ridge, and a broken sensor are obvious before anyone opens a table.</p>
      <div class="swatches">
        <span style="background:#2F9E78">Good</span>
        <span style="background:#E0B83A">Moderate</span>
        <span style="background:#E08A3C">Sensitive</span>
        <span style="background:#D4534A">Unhealthy</span>
        <span style="background:#8E4A86">Very unhealthy</span>
        <span style="background:#6E2434;color:#F6F1E8">Hazardous</span>
        <span style="background:#D4CFC6">Sensor offline</span>
      </div>
      <div class="stats">
        <div class="stat"><b>{f["n_readings"]}</b><span>hourly readings</span></div>
        <div class="stat"><b>{f["n_alerts"]}</b><span>alert episodes</span></div>
        <div class="stat"><b>147</b><span>the seed that rebuilds every row</span></div>
      </div>
      <p class="disclaimer">Synthetic Metro Manila Airshed. Not a DENR-EMB observation, and not a regulatory AQI. Aasen · Eskwelabs EIF fellowship.</p>
    </section>
    """)
    pages.append(sheet(f"""
      <p class="kicker">The assignment</p>
      <h2>A dashboard has to know which bad hour is real.</h2>
      <p class="lede">SD-147 is a difficulty-1 synthetic dataset for an urban-services analyst. The job is to decide where the air is bad, when it is bad, and which hours should not be trusted. The pictures are the point. The tables exist so the pictures can be checked.</p>
      <div class="columns">
        <div class="main">
          <p>Five tables describe one airshed. Coordinates are approximate landmark locations, close enough that a map has Metro Manila’s shape. The operator on every station is Synthetic Metro Manila Airshed, on purpose.</p>
          <p>Run All on <em>SD-147_urban_air_quality.ipynb</em> rewrites the CSVs from seed 147 and stops if a sanity check fails. Every coefficient that is only summarized here is written out in <em>assumptions.md</em>.</p>
          <div class="pull"><p>Roadside EDSA and upland Antipolo share the year and do not share a color.</p></div>
        </div>
        <div class="side">
          <div class="card"><h3>Stations · 12</h3><p>One row per site. Type, city, elevation, and a lat-lon that will sit still on a map.</p></div>
          <div class="card"><h3>Pollutants · 6</h3><p>PM2.5, PM10, NO2, ozone, SO2, and CO, each with a unit and a reason it would move.</p></div>
          <div class="card"><h3>Readings · {f["n_readings"]}</h3><p>Station × pollutant × hour. Quality is ok, telemetry_gap, or sensor_fault.</p></div>
          <div class="card"><h3>Weather · {f["n_weather"]}</h3><p>Temperature, humidity, wind, rain, and pressure at the same hours.</p></div>
          <div class="card"><h3>Alerts · {f["n_alerts"]}</h3><p>An episode opens when the dashboard index reaches 101, and it records what drove it.</p></div>
        </div>
      </div>
    """))
    pages.append(sheet(f"""
      <p class="kicker">The airshed</p>
      <h2>Industry and the big roads hold the year. The ridge does not.</h2>
      <p>Annual mean PM2.5, using hours flagged ok. Marker-sized alert counts in the map later tell the same story: Valenzuela, EDSA, and Marikina’s valley work hardest. Antipolo, about 200 metres up, finishes at {f["ant_pm"]} µg/m³ against EDSA’s {f["edsa_pm"]}.</p>
      <table>
        <thead><tr><th>Station</th><th>City</th><th>Type</th><th style="text-align:right">PM2.5</th><th></th><th style="text-align:right">Alerts</th></tr></thead>
        <tbody>{f["stations_html"]}</tbody>
      </table>
      <p class="small" style="margin-top:3mm">Means are micrograms per cubic metre. The bar is the same annual mean, scaled to 42. Alerts are episodes, not hours.</p>
    """))
    pages.append(sheet("""
      <p class="kicker">The process</p>
      <h2>Every concentration is lognormal, with a memory.</h2>
      <p class="lede">Under the log is a station baseline, then a few shared drivers, then two slow shocks so a dirty hour is usually followed by another one. Neighbouring hours look like weather, not like white noise.</p>
      <div class="columns">
        <div class="main">
          <pre>log PM2.5  =  log(baseline)
            +  0.30 × dry season
            +  0.38 × traffic
            −  0.20 × log(1 + rain)
            −  0.10 × wind, capped at 6 m/s
            +  a regional bad-air day
            +  a shock local to the station</pre>
          <p>Traffic is two bells, at 08:00 and 18:30, with a night floor. Saturday multiplies it by 0.84, Sunday by 0.68, a public holiday by 0.55. The dry-season cosine peaks in early February. Rain arrives in wet-season clusters. Wind disperses, and then stops helping once it is already blowing hard. The cap is there so a typhoon does not drive the instrument to its floor.</p>
        </div>
        <div class="side">
          <div class="card"><h3>PM10</h3><p>PM2.5 plus coarse dust. After every edit it is forced to at least 1.04 × the fine fraction. The ratio is what breaks during construction.</p></div>
          <div class="card"><h3>NO2 and CO</h3><p>The traffic gases. NO2 follows the commute with coefficient 0.62. CO is the same shape, not a second story.</p></div>
          <div class="card"><h3>Ozone</h3><p>Rises toward 14:00 and is cut where fresh NO2 is high. That is why Antipolo is the ozone station and EDSA is the particle station.</p></div>
          <div class="card"><h3>SO2</h3><p>Mostly the baseline. The port and Valenzuela carry it. Traffic only nudges it.</p></div>
        </div>
      </div>
      <p class="small">About 0.4% of the hours that would otherwise be valid are ordinary telemetry gaps, missing independently by pollutant. Reported precision is one decimal, two for CO. Physical clips keep every species inside a range a real instrument could print.</p>
    """))
    pages.append(sheet(f"""
      <p class="kicker">One year, two sites</p>
      <h2>The dry months are orange at the roadside and green on the ridge.</h2>
      <img class="full" src="figures/calendars.png" alt="Calendars of daily PM2.5 at EDSA and Antipolo">
      <p>Color is the daily mean of PM2.5, painted with the US EPA 24-hour breakpoints. Good runs through 12.0 µg/m³, moderate through 35.4, unhealthy for sensitive groups through 55.4. Grey is not a clean day. It is a day whose PM channel was frozen for at least twelve hours.</p>
      <p>On the days EDSA was actually reporting, the median day is {f["edsa_med"]} µg/m³. {f["edsa_above"]}% of those days sit above the moderate breakpoint, and {f["edsa_high"]}% sit above 55.4. The worst day reaches {f["edsa_max"]}. Antipolo’s median day is {f["ant_med"]}, and its worst day is {f["ant_max"]}. The teal stripe in September is the typhoon. The red edges of the year are fireworks.</p>
    """))
    pages.append(sheet(f"""
      <p class="kicker">Where the year was dirtiest</p>
      <h2>The same mean, once as a map and once as a ranking.</h2>
      <img class="full" src="figures/map.png" alt="Station map and ranked annual PM2.5">
      <p>Color is the annual mean. On the map, a larger marker opened more alert episodes. Valenzuela leads at {f["val_pm"]} µg/m³, EDSA is next at {f["edsa_pm"]}, and the episodes pile up on those same roads: {f["top_alerts"]}. {f["ant_alert"]} Nothing in the north of the basin is as clean as the ridge, and nothing on the ridge looks like EDSA.</p>
    """))
    pages.append(sheet(f"""
      <p class="kicker">The shape of a day</p>
      <h2>Particles belong to the commute. Ozone belongs to the afternoon.</h2>
      <img class="full" src="figures/diurnal.png" alt="Diurnal curves for PM2.5, NO2, and ozone">
      <p>The line is the median hour and the band is the middle half of hours. At the roadside, PM2.5 and NO2 rise for the morning peak and again after 18:00. Sunday cuts roadside NO2 from {f["weekday_no2"]} to {f["sunday_no2"]} µg/m³, a {f["sunday_cut"]}% drop across EDSA, Makati, Pasig, and NAIA. Both peaks nearly disappear at Antipolo.</p>
      <p>Ozone does the opposite. Across the airshed the 14:00 mean is {f["o3_14"]} µg/m³ against {f["o3_6"]} at 06:00. On roadside hours from 10:00 to 16:00, ozone and NO2 correlate at {f["r_o3"]}. Fresh exhaust titrates the ozone that sunlight is trying to make.</p>
    """))
    pages.append(sheet("""
      <p class="kicker">Dry season, then the wash</p>
      <h2>January through April is the dark block.</h2>
      <img class="full" src="figures/season.png" alt="Monthly mean PM2.5 by station">
      <p class="small">August at EDSA is only the hours the sensor was reporting. The frozen week is left out of the mean, and the calendar paints those days grey instead.</p>
    """))
    pages.append(sheet(f"""
      <p class="kicker">Rain, then wind</p>
      <h2>A wet day is cleaner. Past a stiff breeze, more wind stops helping.</h2>
      <img class="full" src="figures/washout.png" alt="PM2.5 against daily rain and wind">
      <p>A dry day across the twelve stations sits near {f["dry"]} µg/m³. The bars step down as the day gets wetter, and a day with 40 mm or more sits near {f["heavy"]}. The daily correlation of PM2.5 with rainfall is {f["r_rain"]}, and with wind it is {f["r_wind"]}. PM2.5 and PM10 otherwise move together at r = {f["r_pm"]}, which is the correlation the construction week is designed to break.</p>
      <div class="note"><p>Marikina is the local exception the model allows. On a calm night, wind under 1.6 m/s between 20:00 and 06:00, its PM2.5 log-mean is lifted by 0.18. That is a stand-in for drainage flow into the valley, and it is why a residential site can finish the year beside the industrial ones.</p></div>
    """))
    pages.append(sheet(f"""
      <p class="kicker">Written in, not drawn</p>
      <h2>Four events are not tails of the distribution.</h2>
      <img class="full" src="figures/episodes.png" alt="Four planted edge cases">
      <ol class="edges">
        <li><b>01</b><p><strong>Fireworks, fine mode.</strong> 1 January 00:00–05:00, the evening of 10 February, and the evening of 31 December. At EDSA the New Year window averages {f["spike"]} µg/m³ of PM2.5, against a quiet January week whose median hour is {f["quiet"]}. PM10 is lifted less, so the fine fraction dominates.</p></li>
        <li><b>02</b><p><strong>Construction dust at Valenzuela.</strong> Weekdays 4–15 March, 07:00–17:00. PM2.5 is only nudged, by 1.18×. PM10 gains about 115 µg/m³. The gases are untouched, and the PM10/PM2.5 median goes from {f["ratio_base"]} to {f["ratio_work"]}. Weekends rest.</p></li>
        <li><b>03</b><p><strong>A frozen sensor at EDSA.</strong> 12–18 August, {f["freeze_hours"]} hours, one repeated PM2.5 value of {f["freeze_value"]} µg/m³, flagged sensor_fault. PM10 is frozen with it. Ozone and NO2 keep reporting. Alerts ignore the frozen channels. {f["fault_days"]} days on the calendar go grey.</p></li>
        <li><b>04</b><p><strong>A typhoon, unnamed on purpose.</strong> 15 September 06:00 through 17 September 18:00, every station. Rain, wind, a cooler wetter air mass, and a further cut to every pollutant. EDSA’s PM2.5 mean in that window is {f["typhoon"]} µg/m³, against {f["early"]} in the first ten days of September.</p></li>
      </ol>
    """))
    pages.append(sheet(f"""
      <p class="kicker">The index</p>
      <h2>{f["n_alerts"]} episodes. Most of them are particles, and they last.</h2>
      <div class="columns">
        <div class="main">
          <p>An alert opens when the dashboard index reaches 101 and closes when it falls back, or when the clock jumps by more than an hour and a half. The median episode runs {f["median_h"]} hours. The longest holds for {f["longest_days"]} days at {f["longest_where"]}: a daily mean that rarely slips under the breakpoint, not a stuck sensor. The highest index in the year is {f["peak"]}, at {f["peak_where"]} on {f["peak_when"]}.</p>
          <p>PM2.5 drives {f["pm_share"]}% of episodes. By pollutant: {f["drivers"]}. {f["ozone_note"]}</p>
          <p>{f["levels"]} The index never prints a hazardous day. Particles are judged on a trailing 24-hour mean, with at least 18 hours present. Ozone and CO use eight hours. NO2 and SO2 use the hour itself.</p>
          <div class="pull"><p>This is a consistent dashboard index. It is the US EPA table, not DENR’s, and a partly filled window will overstate a legal AQI.</p></div>
        </div>
        <div class="side">
          <img class="drivers" src="figures/drivers.png" alt="Alert episodes by driver pollutant">
        </div>
      </div>
    """))
    pages.append(sheet(f"""
      <p class="kicker">What the year will sit still for</p>
      <h2>Sixteen checks pass. A few uses will still break it.</h2>
      <p>The notebook refuses to finish unless concentrations stay non-negative, PM10 stays above PM2.5, the correlations point the right way, Sunday is quieter, the ridge is cleaner, and each planted event is still where it was put. The numbers above are those checks, read back from the CSVs rather than from the generator’s hopes.</p>
      <div class="columns">
        <div class="main">
          <div class="card"><h3>Ask this of the tables</h3><p>Which stations and months sit above the 24-hour PM2.5 breakpoints, and how much of that is dry-season traffic. How far Sunday falls below a weekday. Whether rain and wind actually clear the basin once you look at days rather than hours. How long alerts last, and when the driver is particles rather than NO2.</p></div>
          <div class="card"><h3>Three traps, left in on purpose</h3><p>Firework nights that look like combustion. A construction site that inflates PM10 only. A week in August at EDSA where the PM channel is a flat line. Drop the quality flag and that week becomes a clean week that never happened.</p></div>
        </div>
        <div class="side">
          <div class="card"><h3>Left out</h3><p>No chemistry beyond the NO2 titration of ozone. No street canyon, no sea-breeze clock, no transboundary smoke except the three firework nights. Instrument drift, other than one frozen week and the random gaps, is not in the model.</p></div>
          <div class="note"><p>{f["n_gaps"]} readings ({f["gap_pct"]}%) are telemetry gaps. They are missing on purpose and should stay missing in any average.</p></div>
        </div>
      </div>
      <p class="end quiet">Rebuild it with Run All. Seed 147 reproduces the files. Changing the seed changes every row, including the weather the edge cases sit on. The generator is src/aq_model.py. The full specification is assumptions.md.</p>
    """))
    body = "\n".join(pages)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>SD-147 · A year of Metro Manila air</title><style>"
        + REPORT_CSS
        + "</style></head><body>"
        + body
        + "</body></html>"
    )


def slides_html(f: dict[str, str]) -> str:
    slides = f"""
    <section class="slide">
      <p class="kicker">Eskwelabs EIF · Synthetic dataset · SD-147</p>
      <h1>A year of<br>Metro Manila air.</h1>
      <p class="deck">Twelve stations, every hour of 2024. Generated so a bad day, a clean ridge, and a broken sensor are obvious before anyone opens a table.</p>
      <p class="sub">Not a DENR observation · not a regulatory AQI · seed 147</p>
    </section>
    <section class="slide">
      <p class="kicker">What was built</p>
      <h2>Five tables, one airshed.</h2>
      <div class="pills">
        <div class="pill"><em>Stations</em><b>12</b><span>Landmark sites, from the port to the ridge.</span></div>
        <div class="pill"><em>Pollutants</em><b>6</b><span>PM2.5, PM10, NO2, ozone, SO2, CO.</span></div>
        <div class="pill"><em>Readings</em><b>{f["n_readings"]}</b><span>Every station, every pollutant, every hour.</span></div>
        <div class="pill"><em>Weather</em><b>{f["n_weather"]}</b><span>The rain and wind that move the particles.</span></div>
        <div class="pill"><em>Alerts</em><b>{f["n_alerts"]}</b><span>Episodes where the index reached 101.</span></div>
      </div>
      <p class="caption">The audience is an urban-services analyst who has to decide which bad hour is real.</p>
    </section>
    <section class="slide">
      <p class="kicker">How a concentration is made</p>
      <h2>A baseline, a commute, a season, and a memory.</h2>
      <div class="forces">
        <div class="force"><h3>Traffic</h3><p>Bells at 08:00 and 18:30. Saturday is quieter, Sunday quieter still, and a holiday cuts the wave to 0.55.</p></div>
        <div class="force"><h3>Dry season</h3><p>A cosine that peaks in early February. January through April is the dirty block on every roadside row.</p></div>
        <div class="force"><h3>Rain and wind</h3><p>Rain washes particles out. Wind disperses them, then flattens. A typhoon is not allowed to hit the instrument floor.</p></div>
        <div class="force"><h3>Memory</h3><p>Two autoregressive shocks, one shared by the airshed and one local, so a dirty hour is usually followed by another.</p></div>
      </div>
      <p class="aside">Ozone does the opposite of the traffic gases. It rises toward 14:00, and it falls wherever fresh NO2 is high.</p>
    </section>
    <section class="slide">
      <p class="kicker">One year, two sites</p>
      <h2>Same calendar. Not the same color.</h2>
      <img class="fig" src="figures/calendars.png" alt="EDSA and Antipolo calendars">
      <p class="caption">EDSA’s median reporting day is {f["edsa_med"]} µg/m³, and {f["edsa_above"]}% of those days sit above moderate. Antipolo’s median day is {f["ant_med"]}. Grey is a frozen sensor. Teal is the typhoon.</p>
    </section>
    <section class="slide">
      <p class="kicker">Where the year was dirtiest</p>
      <h2>The roads and the plants. Then a long drop to the ridge.</h2>
      <img class="fig" src="figures/map.png" alt="Map and ranking">
      <p class="caption">Valenzuela {f["val_pm"]}, EDSA {f["edsa_pm"]}, Antipolo {f["ant_pm"]} µg/m³. Larger markers opened more episodes. {f["ant_alert"]}</p>
    </section>
    <section class="slide">
      <p class="kicker">The shape of a day</p>
      <h2>Commute, Sunday, afternoon.</h2>
      <img class="fig" src="figures/diurnal.png" alt="Diurnal curves">
      <p class="caption">Sunday roadside NO2 falls from {f["weekday_no2"]} to {f["sunday_no2"]} µg/m³. Ozone at 14:00 is {f["o3_14"]} against {f["o3_6"]} at 06:00, and it opposes NO2 at r = {f["r_o3"]}.</p>
    </section>
    <section class="slide">
      <p class="kicker">The dry season</p>
      <h2>January through April, on every roadside row.</h2>
      <img class="fig" src="figures/season.png" alt="Monthly grid">
      <p class="caption">The wet months cool the grid. August at EDSA is only the hours the sensor was actually reporting.</p>
    </section>
    <section class="slide">
      <p class="kicker">What clears the basin</p>
      <h2>Rain first. Wind second. Then wind stops helping.</h2>
      <img class="fig" src="figures/washout.png" alt="Washout">
      <p class="caption">A dry day sits near {f["dry"]} µg/m³. A day with 40 mm or more sits near {f["heavy"]}. Daily r with rain is {f["r_rain"]}, with wind {f["r_wind"]}.</p>
    </section>
    <section class="slide">
      <p class="kicker">Not a random tail</p>
      <h2>Four events were written in.</h2>
      <div class="split">
        <img src="figures/episodes.png" alt="Edge cases">
        <ol>
          <li><strong>Fireworks</strong><span>EDSA, New Year morning, averages {f["spike"]} µg/m³. A quiet week’s median hour is {f["quiet"]}.</span></li>
          <li><strong>Valenzuela dust</strong><span>PM10/PM2.5 goes from {f["ratio_base"]} to {f["ratio_work"]}. The gases do not move. Weekends rest.</span></li>
          <li><strong>Frozen sensor</strong><span>{f["freeze_hours"]} hours at EDSA stuck on {f["freeze_value"]} µg/m³, flagged. Alerts ignore it.</span></li>
          <li><strong>Typhoon</strong><span>EDSA falls to {f["typhoon"]} µg/m³, from {f["early"]} in early September. Not a named storm.</span></li>
        </ol>
      </div>
    </section>
    <section class="slide">
      <p class="kicker">Alerts</p>
      <h2>Episodes, not single bad hours.</h2>
      <div class="nums">
        <div class="num"><b>{f["n_alerts"]}</b><span>episodes opened once the index reached 101.</span></div>
        <div class="num"><b>{f["median_h"]} h</b><span>median length. The longest, at Valenzuela, holds for {f["longest_days"]} days.</span></div>
        <div class="num"><b>{f["pm_share"]}%</b><span>are driven by PM2.5. Ozone opens none. An eight-hour mean keeps the afternoon spike from qualifying.</span></div>
        <div class="num"><b>{f["peak"]}</b><span>highest index, at {f["peak_where"]} on {f["peak_when"]}. Unhealthy. Never hazardous.</span></div>
      </div>
    </section>
    <section class="slide">
      <p class="kicker">Read it with the flag on</p>
      <h2>Three ways to lie with this file.</h2>
      <div class="limits">
        <div><b>01</b><p>Call it DENR data. It is a synthetic airshed with landmark coordinates.</p></div>
        <div><b>02</b><p>Call the index a legal AQI. The breakpoints are the US EPA table, on trailing means, not DENR’s.</p></div>
        <div><b>03</b><p>Average August at EDSA without the quality flag. The frozen week becomes a clean week that never happened.</p></div>
      </div>
      <p class="caption">Firework nights look like combustion. The Valenzuela job inflates PM10 only. Both are features.</p>
    </section>
    <section class="slide">
      <p class="kicker">What to take from it</p>
      <h2>The basin has a season. The instrument has a week.</h2>
      <ul class="asks">
        <li>Which bad months are dry-season traffic, and which days are the sensor.</li>
        <li>How far a Sunday, a rain day, and a ridge station sit below EDSA.</li>
        <li>When an alert is particles, and when it is the evening NO2 peak.</li>
      </ul>
      <p class="sub">Seed 147 rebuilds every row · notebook, assumptions.md, five CSVs</p>
    </section>
    """
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>SD-147 slides</title><style>"
        + SLIDE_CSS
        + "</style></head><body>"
        + slides
        + "</body></html>"
    )


def print_pdf(html: Path, pdf: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if pdf.exists():
        pdf.unlink()
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        ],
        check=True,
        timeout=120,
    )


def main() -> None:
    tables = load()
    info = facts(tables)
    save_figures(tables)
    report = OUT / "report.html"
    slides = OUT / "slides.html"
    report.write_text(report_html(info), encoding="utf-8")
    slides.write_text(slides_html(info), encoding="utf-8")
    print_pdf(report, OUT / "SD-147_Urban_Air_Quality_Report.pdf")
    print_pdf(slides, OUT / "SD-147_Urban_Air_Quality_Slides.pdf")
    for key in (
        "n_readings", "n_alerts", "edsa_pm", "ant_pm", "val_pm", "edsa_med", "edsa_above",
        "edsa_high", "edsa_max", "ant_med", "ant_max", "spike", "quiet", "typhoon", "early",
        "ratio_work", "ratio_base", "freeze_value", "median_h", "peak", "peak_where",
        "peak_when", "pm_share", "drivers", "levels", "dry", "wet", "r_rain", "r_wind",
        "sunday_no2", "weekday_no2", "o3_14", "o3_6", "r_o3", "top_alerts", "ant_episodes",
    ):
        print(f"{key}: {info[key]}")


if __name__ == "__main__":
    main()
