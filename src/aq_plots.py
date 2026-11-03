"""Figures for the SD-147 notebook. Each function returns a matplotlib Figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

PAPER = "#F6F1E8"
INK = "#1C1917"
MUTED = "#78716C"
LINE = "#E4DDD2"
TEAL = "#0F6E6B"
CLAY = "#C2412D"

AQI_COLORS = ["#2F9E78", "#E0B83A", "#E08A3C", "#D4534A", "#8E4A86", "#6E2434"]
AQI_BOUNDS = [0, 12.1, 35.5, 55.5, 150.5, 250.5, 500]
AQI_NAMES = ["Good", "Moderate", "Sensitive", "Unhealthy", "Very unhealthy", "Hazardous"]
FAULT = "#D4CFC6"

PM_CMAP = LinearSegmentedColormap.from_list(
    "pm", ["#7ECFC2", "#E4C56A", "#E07A3C", "#B84332", "#6E2434"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": "#D9D1C5",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": "#44403C",
        "ytick.color": "#44403C",
        "text.color": INK,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 140,
    })


def _footer(fig, extra: str = "") -> None:
    note = "SD-147  ·  synthetic Metro Manila airshed  ·  2024"
    if extra:
        note = f"{note}  ·  {extra}"
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)


def _daily(readings: pd.DataFrame, station_id: str) -> pd.DataFrame:
    sub = readings[(readings["station_id"] == station_id) & (readings["pollutant_id"] == "PM25")].copy()
    sub["day"] = sub["observed_at"].dt.floor("D")
    rows = []
    for day, block in sub.groupby("day"):
        fault_hours = int((block["quality_flag"] == "sensor_fault").sum())
        ok = block.loc[block["quality_flag"] == "ok", "value"]
        rows.append({
            "day": day,
            "mean": ok.mean() if len(ok) else np.nan,
            "fault": fault_hours >= 12,
        })
    return pd.DataFrame(rows)


def calendars(readings: pd.DataFrame) -> plt.Figure:
    """EDSA and Antipolo, one year, same color language."""
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 5.5))
    titles = [
        ("STA-EDSA", "EDSA–Guadalupe  ·  a roadside year"),
        ("STA-ANT", "Antipolo Upland  ·  the same year, above the basin"),
    ]
    cmap = ListedColormap(AQI_COLORS)
    norm = BoundaryNorm(AQI_BOUNDS, cmap.N)
    for ax, (sid, title) in zip(axes, titles):
        daily = _daily(readings, sid)
        daily["week"] = ((daily["day"] - pd.Timestamp("2024-01-01")).dt.days // 7).astype(int)
        daily["dow"] = daily["day"].dt.dayofweek
        grid = np.full((7, int(daily["week"].max()) + 1), np.nan)
        fault = np.zeros_like(grid, dtype=bool)
        for row in daily.itertuples(index=False):
            grid[row.dow, row.week] = row.mean
            fault[row.dow, row.week] = bool(row.fault)
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ys, xs = np.where(fault)
        if len(xs):
            ax.scatter(xs, ys, marker="s", s=64, c=FAULT, zorder=3)
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        months = daily.groupby(daily["day"].dt.month)["week"].min()
        ax.set_xticks(months.to_numpy())
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", pad=6)
    handles = [Patch(facecolor=c, edgecolor="none", label=name) for c, name in zip(AQI_COLORS, AQI_NAMES)]
    handles.append(Patch(facecolor=FAULT, edgecolor="none", label="Sensor offline"))
    fig.legend(handles=handles, ncol=7, frameon=False, loc="lower center", fontsize=8.5, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.01, 0.012, "SD-147  ·  synthetic Metro Manila airshed  ·  2024  ·  daily-mean PM2.5, 24-hour breakpoints", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig


def station_map(stations: pd.DataFrame, readings: pd.DataFrame, alerts: pd.DataFrame) -> plt.Figure:
    annual = (
        readings[(readings["pollutant_id"] == "PM25") & (readings["quality_flag"] == "ok")]
        .groupby("station_id")["value"].mean()
    )
    episodes = alerts.groupby("station_id").size()
    frame = stations.set_index("station_id").join(annual.rename("pm25")).join(episodes.rename("episodes"))
    frame = frame.sort_values("pm25", ascending=True)
    norm = plt.Normalize(8, 42)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 6.3), gridspec_kw={"width_ratios": [1.05, 0.95]}
    )
    ax.set_aspect(1 / np.cos(np.deg2rad(14.55)))
    ax.scatter(
        frame["longitude"], frame["latitude"],
        c=frame["pm25"], s=180 + frame["episodes"] * 0.85,
        cmap=PM_CMAP, norm=norm,
        edgecolor=INK, linewidth=0.8, zorder=3,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("The airshed", loc="left")
    ax.set_xlim(120.93, 121.28)
    ax.set_ylim(14.41, 14.74)

    colors = PM_CMAP(norm(frame["pm25"].to_numpy()))
    ax2.barh(frame["station_name"], frame["pm25"], color=colors, height=0.72, zorder=2)
    for name, value in zip(frame["station_name"], frame["pm25"]):
        ax2.text(value + 0.5, name, f"{value:.0f}", va="center", ha="left", fontsize=8, color=INK)
    ax2.set_xlim(0, 52)
    ax2.set_xlabel("Annual mean PM2.5 (µg/m³)")
    ax2.set_title("Same stations, ranked", loc="left")
    ax2.tick_params(length=0)
    _footer(fig, "larger markers opened more alert episodes")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def diurnal(readings: pd.DataFrame) -> plt.Figure:
    ok = readings[readings["quality_flag"] == "ok"].copy()
    ok["hour"] = ok["observed_at"].dt.hour
    ok["dow"] = ok["observed_at"].dt.dayofweek
    specs = [
        ("PM25", "PM2.5", "µg/m³"),
        ("NO2", "Nitrogen dioxide", "µg/m³"),
        ("O3", "Ozone", "µg/m³"),
    ]
    series = {
        "EDSA weekday": ((ok["station_id"] == "STA-EDSA") & (ok["dow"] < 5), INK, 2.2),
        "EDSA Sunday": ((ok["station_id"] == "STA-EDSA") & (ok["dow"] == 6), TEAL, 1.8),
        "Antipolo weekday": ((ok["station_id"] == "STA-ANT") & (ok["dow"] < 5), CLAY, 1.8),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.15))
    for ax, (pid, name, unit) in zip(axes, specs):
        for label, (mask, color, width) in series.items():
            block = ok.loc[mask & (ok["pollutant_id"] == pid)]
            band = block.groupby("hour")["value"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(band.index, band[0.25], band[0.75], color=color, alpha=0.13, linewidth=0)
            ax.plot(band.index, band[0.5], color=color, lw=width, label=label)
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_title(name, loc="left")
        ax.set_xlabel("Hour")
        ax.set_ylabel(unit)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("The shape of a day", x=0.01, ha="left", fontsize=14)
    _footer(fig, "line is the median, band is the middle half of hours")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    return fig


def episodes(readings: pd.DataFrame, weather: pd.DataFrame) -> plt.Figure:
    def pull(station, pollutant, start, end):
        block = readings[
            (readings["station_id"] == station)
            & (readings["pollutant_id"] == pollutant)
            & (readings["observed_at"].between(start, end))
        ].sort_values("observed_at")
        return block

    panels = [
        ("New Year fireworks", "STA-EDSA", "PM25", "2024-01-01", "2024-01-04 23:00", "2024-01-01", "2024-01-01 05:00", "PM2.5"),
        ("Valenzuela dust", "STA-VAL", "PM10", "2024-02-26", "2024-03-22", "2024-03-04", "2024-03-15 23:00", "PM10"),
        ("EDSA sensor freeze", "STA-EDSA", "PM25", "2024-08-05", "2024-08-25", "2024-08-12", "2024-08-18 23:00", "PM2.5"),
        ("Typhoon passage", "STA-EDSA", "PM25", "2024-09-08", "2024-09-24", "2024-09-15 06:00", "2024-09-17 18:00", "PM2.5"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12.4, 8.4))
    for ax, (title, station, pollutant, start, end, shade_a, shade_b, ylab) in zip(axes, panels):
        block = pull(station, pollutant, start, end)
        ax.axvspan(pd.Timestamp(shade_a), pd.Timestamp(shade_b), color="#E7D3A1", alpha=0.55, lw=0)
        ok = block["quality_flag"] == "ok"
        fault = block["quality_flag"] == "sensor_fault"
        if pollutant == "PM10":
            fine = pull(station, "PM25", start, end)
            fine_ok = fine["quality_flag"] == "ok"
            ax.plot(fine.loc[fine_ok, "observed_at"], fine.loc[fine_ok, "value"], color=TEAL, lw=1.15, label="PM2.5")
            ax.plot(block.loc[ok, "observed_at"], block.loc[ok, "value"], color=INK, lw=1.15, label="PM10")
            ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
            ax.set_ylabel("µg/m³")
        else:
            ax.plot(block.loc[ok, "observed_at"], block.loc[ok, "value"], color=INK, lw=1.15)
            if fault.any():
                ax.plot(block.loc[fault, "observed_at"], block.loc[fault, "value"], color=CLAY, lw=1.6)
            ax.set_ylabel(f"{ylab} µg/m³")
        ax.set_title(title, loc="left", fontsize=11)
        ax.margins(x=0.01)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("")
    _footer(fig, "gold band is the deliberate edge case")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def season_grid(readings: pd.DataFrame, stations: pd.DataFrame) -> plt.Figure:
    ok = readings[(readings["pollutant_id"] == "PM25") & (readings["quality_flag"] == "ok")].copy()
    ok["month"] = ok["observed_at"].dt.month
    grid = ok.pivot_table(index="station_id", columns="month", values="value", aggfunc="mean")
    order = stations.sort_values("latitude", ascending=False)["station_id"]
    grid = grid.loc[order]
    names = stations.set_index("station_id")["station_name"]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    im = ax.imshow(grid.to_numpy(), cmap=PM_CMAP, aspect="auto", vmin=8, vmax=48)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(names.loc[grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.iloc[i, j]
            color = "white" if val >= 36 else INK
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Monthly mean PM2.5, stations ordered north to south", loc="left")
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    bar.outline.set_visible(False)
    bar.set_label("µg/m³")
    _footer(fig, "August at EDSA is the weeks the sensor was actually reporting")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def washout(readings: pd.DataFrame, weather: pd.DataFrame) -> plt.Figure:
    pm = readings[(readings["pollutant_id"] == "PM25") & (readings["quality_flag"] == "ok")].copy()
    pm["day"] = pm["observed_at"].dt.floor("D")
    daily_pm = pm.groupby(["station_id", "day"])["value"].mean().rename("pm25")
    rain = weather.copy()
    rain["day"] = rain["observed_at"].dt.floor("D")
    daily_rain = rain.groupby(["station_id", "day"])["rainfall_mm"].sum().rename("rain")
    daily_wind = rain.groupby(["station_id", "day"])["wind_speed_ms"].mean().rename("wind")
    joined = pd.concat([daily_pm, daily_rain, daily_wind], axis=1).dropna().reset_index()

    bins = [0, 0.1, 1, 5, 15, 40, 200]
    labels = ["Dry", "<1 mm", "1–5", "5–15", "15–40", "40+"]
    joined["bucket"] = pd.cut(joined["rain"], bins=bins, labels=labels, right=False, include_lowest=True)
    summary = joined.groupby("bucket", observed=True)["pm25"].agg(["mean", "median", "count"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    norm = plt.Normalize(18, 36)
    colors = [PM_CMAP(norm(v)) for v in summary["mean"]]
    bars = ax.bar(summary.index.astype(str), summary["mean"], color=colors, width=0.72, zorder=2)
    for bar, med, mean in zip(bars, summary["median"], summary["mean"]):
        tick = "white" if mean >= 24 else INK
        ax.plot(
            [bar.get_x() + 0.18, bar.get_x() + bar.get_width() - 0.18],
            [med, med], color=tick, lw=1.6, zorder=3,
        )
    ax.set_ylabel("Daily mean PM2.5 (µg/m³)")
    ax.set_xlabel("Rain that day")
    ax.set_title("A wet day is a cleaner day", loc="left")
    ax.set_ylim(0, summary["mean"].max() * 1.25)

    ax2 = axes[1]
    sample = joined.sample(900, random_state=147)
    ax2.scatter(sample["wind"], sample["pm25"], s=10, c=TEAL, alpha=0.18, linewidths=0)
    wind_bins = pd.cut(joined["wind"], bins=[0.5, 1.5, 2.5, 3.5, 5, 8, 16])
    wind_mean = joined.groupby(wind_bins, observed=True)["pm25"].mean()
    centers = [interval.mid for interval in wind_mean.index]
    ax2.plot(centers, wind_mean.to_numpy(), color=CLAY, lw=2.2, marker="o", ms=5)
    ax2.set_xlabel("Daily mean wind (m/s)")
    ax2.set_ylabel("Daily mean PM2.5 (µg/m³)")
    ax2.set_title("Wind finishes the job", loc="left")
    _footer(fig, "white tick on each bar is the median")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
