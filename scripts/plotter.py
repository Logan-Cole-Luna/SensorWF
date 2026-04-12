"""
plotter.py — All visualisations for SLL telemetry analysis.

Each function saves one figure to output_dir and returns the file path.

Sections
--------
  01–15  General telemetry time-series and distributions
  16–20  ADCS sensor plots (conditional on active data)
  E2–E3, E5  ML evaluation plots (metrics bars, confusion matrices, latency)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

# ── Shared theme ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE     = sns.color_palette("tab10")
WIDE        = (14, 5)
TALL        = (14, 8)
TEMP_COLS   = ["TEMP_SOLAR","TEMP_EPS","TEMP_BATTERY","TEMP_BACKPLANE",
               "TEMP_OBDH","TEMP_ADCS","TEMP_WHEEL","TEMP_COMMS","TEMP_PAYLOAD"]
VOLT_COLS   = ["SOLAR_V","SYS_V","PLAT_5V_V","OBC_PWR_V",
               "COMMS_PWR_V","ADCS_5V_V","PAYLOAD_5V_V"]
CURR_COLS   = ["SYS_I","OBC_PWR_I","COMMS_PWR_I","PLAT_5V_I",
               "ADCS_5V_I","PAYLOAD_5V_I","BATT_CHR_I"]


def _save(fig, output_dir: str, filename: str) -> str:
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_series_with_band(
    ax,
    df: pd.DataFrame,
    y_col: str,
    label: str,
    color,
    linewidth: float = 1.8,
    linestyle: str = "-",
    alpha: float = 1.0,
    band_alpha: float = 0.16,
) -> bool:
    """Plot mean line and optional ±1σ band from <col>__std when present."""
    if "elapsed_s" not in df.columns or y_col not in df.columns:
        return False

    x = pd.to_numeric(df["elapsed_s"], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        return False

    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    ax.plot(xv, yv, label=label, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=alpha)

    std_col = f"{y_col}__std"
    if std_col in df.columns:
        s = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0)
        sv = s[mask].to_numpy(dtype=float)
        ax.fill_between(xv, yv - sv, yv + sv, color=color, alpha=band_alpha)
    return True




# ─────────────────────────────────────────────────────────────────────────────
# 01 — Board Temperature Time-Series
# ─────────────────────────────────────────────────────────────────────────────
def plot_board_temperatures(cdh: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, col in enumerate(TEMP_COLS):
        _plot_series_with_band(
            ax, cdh, col, col.replace("TEMP_", ""),
            PALETTE[i % len(PALETTE)], linewidth=1.8
        )
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Board Temperatures — Full Session")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    plt.tight_layout()
    return _save(fig, output_dir, "01_board_temperatures.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — OBDH Thermal Ramp with Linear Fit
# ─────────────────────────────────────────────────────────────────────────────
def plot_obdh_ramp(cdh: pd.DataFrame, output_dir: str) -> str:
    """OBDH thermal ramp with linear fit computed inline."""
    t = cdh["elapsed_s"]
    y = cdh["TEMP_OBDH"]
    valid = y.dropna()
    tv    = t.loc[valid.index]
    slope, intercept, r, *_ = sp_stats.linregress(tv, valid)
    fit  = slope * t + intercept
    rate = slope * 60  # °C/min

    fig, ax = plt.subplots(figsize=WIDE)
    _plot_series_with_band(ax, cdh, "TEMP_OBDH", "TEMP_OBDH", PALETTE[4], linewidth=2.0)
    ax.plot(t, fit, "--", color="crimson", linewidth=2,
            label=f"Linear fit: {rate:.3f} °C/min  (R²={r**2:.4f})")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("OBDH (On-Board Data Handling) Thermal Ramp")
    ax.legend()
    plt.tight_layout()
    return _save(fig, output_dir, "02_obdh_thermal_ramp.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Power Rail Voltages
# ─────────────────────────────────────────────────────────────────────────────
def plot_power_rails(cdh: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=WIDE)
    for i, col in enumerate(VOLT_COLS):
        _plot_series_with_band(ax, cdh, col, col, PALETTE[i % len(PALETTE)], linewidth=1.8)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Power Rail Voltages — Full Session")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "03_power_rails.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Stuck Thermistor Channels
# ─────────────────────────────────────────────────────────────────────────────
def plot_thermistor_anomaly(cdh: pd.DataFrame, output_dir: str) -> str:
    rod_cols  = [c for c in cdh.columns if c.startswith("THERMISTOR_ROD_")]
    bath_cols = [c for c in cdh.columns if "BATH" in c or "EXP3" in c]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for i, col in enumerate(rod_cols[:8]):
        _plot_series_with_band(
            ax, cdh, col, col.replace("THERMISTOR_", ""),
            PALETTE[i % len(PALETTE)], linewidth=1.5, alpha=0.75, band_alpha=0.10
        )
    ax.axhline(77.18, color="red", linestyle="--", linewidth=1.4,
               alpha=0.7, label="Stuck value (77.18 °C)")
    ax.set_title("Thermal Rod Sensors (ROD 1–8) — All Stuck")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=2)

    ax2 = axes[1]
    for i, col in enumerate(bath_cols):
        _plot_series_with_band(
            ax2, cdh, col, col.replace("THERMISTOR_", ""), PALETTE[i % len(PALETTE)]
        )
    _plot_series_with_band(ax2, cdh, "TEMP_OBDH", "TEMP_OBDH (real)", "black", linestyle=":")
    ax2.set_title("Bath/Exp Thermistors vs Real Board Temp")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return _save(fig, output_dir, "04_thermistor_anomaly.png")


# ─────────────────────────────────────────────────────────────────────────────
# 05 — Temperature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_correlation(cdh: pd.DataFrame, output_dir: str) -> str:
    cols   = [c for c in TEMP_COLS if c in cdh.columns]
    corr   = cdh[cols].corr()
    labels = [c.replace("TEMP_", "") for c in cols]
    corr.index = corr.columns = labels

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Board Temperature Correlation Matrix")
    plt.tight_layout()
    return _save(fig, output_dir, "05_temperature_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 06 — Residual Rail Voltages
# ─────────────────────────────────────────────────────────────────────────────
def plot_residual_rails(cdh: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=WIDE)
    for col, label, color in [
        ("ADCS_5V_V",    "ADCS 5V Rail (OFF)",    PALETTE[0]),
        ("PAYLOAD_5V_V", "Payload 5V Rail (OFF)", PALETTE[1]),
    ]:
        if _plot_series_with_band(ax, cdh, col, label, color, linewidth=1.8):
            ax.fill_between(cdh["elapsed_s"], 0, cdh[col], alpha=0.15, color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0,
               alpha=0.5, label="Expected 0 V")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Residual Voltages on Powered-Off Subsystem Rails")
    ax.legend()
    plt.tight_layout()
    return _save(fig, output_dir, "06_residual_rail_voltages.png")


# ─────────────────────────────────────────────────────────────────────────────
# 07 — Summary Dashboard (6 panels)
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(cdh: pd.DataFrame, output_dir: str) -> str:
    """6-panel session dashboard: temps, OBDH ramp, rails, residuals, thermistors, current."""
    valid = cdh["TEMP_OBDH"].dropna()
    tv    = cdh["elapsed_s"].loc[valid.index]
    slope, intercept, r, *_ = sp_stats.linregress(tv, valid)
    rate  = slope * 60
    fit_y = slope * cdh["elapsed_s"] + intercept

    rod_cols = [c for c in cdh.columns if c.startswith("THERMISTOR_ROD_")]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(
        "Satellite Learning Laboratory — Telemetry Dashboard",
        fontsize=13, fontweight="bold",
    )

    # A — All board temps
    ax = axes[0, 0]
    for i, col in enumerate(TEMP_COLS):
        _plot_series_with_band(
            ax, cdh, col, col.replace("TEMP_", ""),
            PALETTE[i % len(PALETTE)], linewidth=1.4, band_alpha=0.10
        )
    ax.set_title("A — Board Temperatures"); ax.set_ylabel("°C")
    ax.legend(fontsize=7, ncol=2)

    # B — OBDH ramp
    ax = axes[0, 1]
    _plot_series_with_band(ax, cdh, "TEMP_OBDH", "TEMP_OBDH", PALETTE[4], linewidth=1.8)
    ax.plot(cdh["elapsed_s"], fit_y, "--", color="crimson", linewidth=2,
            label=f"{rate:.3f} °C/min")
    ax.set_title("B — OBDH Thermal Ramp (Anomaly)"); ax.set_ylabel("°C")
    ax.legend(fontsize=9)

    # C — Primary rails
    ax = axes[1, 0]
    for i, col in enumerate(["SYS_V","PLAT_5V_V","OBC_PWR_V","COMMS_PWR_V"]):
        _plot_series_with_band(ax, cdh, col, col, PALETTE[i], linewidth=1.6)
    ax.set_title("C — Primary Power Rails"); ax.set_ylabel("V")
    ax.legend(fontsize=9)

    # D — Residual rails
    ax = axes[1, 1]
    _plot_series_with_band(ax, cdh, "ADCS_5V_V", "ADCS 5V (OFF)", PALETTE[0], linewidth=1.8)
    _plot_series_with_band(ax, cdh, "PAYLOAD_5V_V", "Payload 5V (OFF)", PALETTE[1], linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("D — Residual Voltages (Anomaly)"); ax.set_ylabel("V")
    ax.legend(fontsize=9)

    # E — Stuck thermistor vs real
    ax = axes[2, 0]
    if rod_cols:
        _plot_series_with_band(ax, cdh, rod_cols[0], f"{rod_cols[0]} (stuck)", "red", linewidth=2)
    _plot_series_with_band(ax, cdh, "TEMP_OBDH", "TEMP_OBDH (real)", "steelblue", linewidth=1.8)
    _plot_series_with_band(ax, cdh, "TEMP_EPS", "TEMP_EPS (real)", "green", linewidth=1.8)
    ax.set_title("E — Stuck Thermistor vs Real Sensors"); ax.set_ylabel("°C")
    ax.legend(fontsize=9)

    # F — Battery charge current
    ax = axes[2, 1]
    if "BATT_CHR_I" in cdh.columns and cdh["BATT_CHR_I"].notna().any():
        _plot_series_with_band(ax, cdh, "BATT_CHR_I", "BATT_CHR_I", "crimson", linewidth=1.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.fill_between(cdh["elapsed_s"], cdh["BATT_CHR_I"], 0,
                        where=cdh["BATT_CHR_I"] < 0, alpha=0.15, color="crimson",
                        label="Discharging")
        ax.fill_between(cdh["elapsed_s"], cdh["BATT_CHR_I"], 0,
                        where=cdh["BATT_CHR_I"] > 0, alpha=0.15, color="green",
                        label="Charging")
    ax.set_title("F — Battery Charge Current (signed)")
    ax.set_ylabel("ADC Count")
    ax.legend(fontsize=8)

    for row in axes:
        for a in row:
            if not a.get_xlabel():
                a.set_xlabel("Elapsed Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_dir, "07_summary_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 08 — Temperature Distribution Violin Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_distributions(cdh: pd.DataFrame, output_dir: str) -> str:
    cols   = [c for c in TEMP_COLS if c in cdh.columns]
    melted = cdh[cols].copy()
    melted.columns = [c.replace("TEMP_", "") for c in cols]
    melted = melted.melt(var_name="Sensor", value_name="Temperature (°C)")

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=melted, x="Sensor", y="Temperature (°C)",
                   hue="Sensor", palette="muted", inner="box",
                   legend=False, ax=ax)
    ax.set_title("Board Temperature Distributions — Full Session")
    ax.set_xlabel("Board / Sensor")
    plt.tight_layout()
    return _save(fig, output_dir, "08_temperature_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 09 — Current Draw per Subsystem (raw ADC counts)
# ─────────────────────────────────────────────────────────────────────────────
def plot_current_draw(cdh: pd.DataFrame, output_dir: str) -> str:
    """Raw ADC current counts for each powered rail."""
    active = [c for c in CURR_COLS if c in cdh.columns and
              c != "BATT_CHR_I" and cdh[c].notna().any() and cdh[c].max() > 0]
    batt   = "BATT_CHR_I"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for i, col in enumerate(active):
        _plot_series_with_band(
            ax, cdh, col, col.replace("_I", ""),
            PALETTE[i % len(PALETTE)], linewidth=1.8
        )
    ax.set_title("Subsystem Current Draw (ADC counts, proportional to mA)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Raw Current (ADC counts)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if batt in cdh.columns and cdh[batt].notna().any():
        _plot_series_with_band(ax2, cdh, batt, "BATT_CHR_I", "crimson", linewidth=1.8)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1.0,
                    alpha=0.6, label="0 (charging/discharging boundary)")
        ax2.fill_between(cdh["elapsed_s"], cdh[batt], 0,
                         where=cdh[batt] < 0, alpha=0.15, color="crimson",
                         label="Discharging")
        ax2.fill_between(cdh["elapsed_s"], cdh[batt], 0,
                         where=cdh[batt] > 0, alpha=0.15, color="green",
                         label="Charging")
    ax2.set_title("Battery Charge Current (signed — negative = discharging)")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Signed ADC Count")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return _save(fig, output_dir, "09_current_draw.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10 — Normalised Temperature Deltas from Baseline (t=0)
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_deltas(cdh: pd.DataFrame, output_dir: str) -> str:
    """Temperature drift from session start — shows OBDH ramp relative to t=0."""
    cols = [c for c in TEMP_COLS if c in cdh.columns]

    fig, ax = plt.subplots(figsize=WIDE)
    for i, col in enumerate(cols):
        baseline = cdh[col].iloc[0]
        delta    = cdh[col] - baseline
        ax.plot(cdh["elapsed_s"], delta,
                label=col.replace("TEMP_", ""), linewidth=1.8,
                color=PALETTE[i % len(PALETTE)])
        std_col = f"{col}__std"
        if std_col in cdh.columns:
            s = pd.to_numeric(cdh[std_col], errors="coerce").fillna(0.0)
            ax.fill_between(cdh["elapsed_s"], delta - s, delta + s,
                            color=PALETTE[i % len(PALETTE)], alpha=0.14)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("ΔTemperature from Baseline (°C)")
    ax.set_title("Temperature Drift from Session Start — Normalised Deltas")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "10_temperature_deltas.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11 — OBDH Rolling Statistics (mean ± 1σ band)
# ─────────────────────────────────────────────────────────────────────────────
def plot_obdh_rolling_stats(cdh: pd.DataFrame, output_dir: str) -> str:
    """
    Rolling 15-sample mean and ±1σ band to show the sustained trend
    and local variability of the OBDH temperature.
    """
    WINDOW = 15
    y      = cdh["TEMP_OBDH"]
    roll_m = y.rolling(WINDOW, center=True, min_periods=3).mean()
    roll_s = y.rolling(WINDOW, center=True, min_periods=3).std()

    fig, ax = plt.subplots(figsize=WIDE)
    ax.scatter(cdh["elapsed_s"], y, s=8, alpha=0.4,
               color=PALETTE[4], zorder=2, label="Raw samples")
    ax.plot(cdh["elapsed_s"], roll_m, color="crimson",
            linewidth=2, zorder=3, label=f"Rolling mean (w={WINDOW})")
    ax.fill_between(cdh["elapsed_s"],
                    roll_m - roll_s, roll_m + roll_s,
                    alpha=0.25, color="crimson", label="±1σ band")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("OBDH Temperature — Rolling Mean and Variability Band")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "11_obdh_rolling_stats.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12 — Packet Inter-Arrival Timing Jitter
# ─────────────────────────────────────────────────────────────────────────────
def plot_packet_timing(cdh: pd.DataFrame, output_dir: str) -> str:
    """
    Histogram + KDE of CDH packet inter-arrival intervals to reveal
    timing irregularities or dropped packets.
    """
    intervals = cdh["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    nominal   = intervals.median()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.histplot(intervals, bins=30, kde=True, ax=ax, color=PALETTE[0])
    ax.axvline(nominal, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median = {nominal:.3f} s")
    ax.set_xlabel("Inter-Arrival Interval (s)")
    ax.set_ylabel("Count")
    ax.set_title("CDH Packet Arrival Interval Distribution")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(range(len(intervals)), intervals.values,
             color=PALETTE[1], linewidth=1.2, alpha=0.8)
    ax2.axhline(nominal, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Median = {nominal:.3f} s")
    # Flag large gaps (> 3× median)
    big_gap_idx = np.where(intervals.values > 3 * nominal)[0]
    if len(big_gap_idx):
        ax2.scatter(big_gap_idx, intervals.values[big_gap_idx],
                    color="red", zorder=5, s=50, label=f"Large gaps (>{3*nominal:.2f} s)")
    ax2.set_xlabel("Packet Index")
    ax2.set_ylabel("Interval (s)")
    ax2.set_title("CDH Packet Timing — Sequential View")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return _save(fig, output_dir, "12_packet_timing_jitter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13 — Temperature Rate-of-Change (dT/dt)
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_roc(cdh: pd.DataFrame, output_dir: str) -> str:
    """
    Numerical first derivative (°C/s) of each board temperature,
    smoothed with a rolling mean to reduce noise.
    """
    SMOOTH = 10
    cols   = [c for c in TEMP_COLS if c in cdh.columns]

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, col in enumerate(cols):
        dt    = cdh["elapsed_s"].diff()
        dtemp = cdh[col].diff()
        roc   = (dtemp / dt).rolling(SMOOTH, center=True, min_periods=2).mean()
        ax.plot(cdh["elapsed_s"], roc,
                label=col.replace("TEMP_", ""), linewidth=1.6,
                color=PALETTE[i % len(PALETTE)])

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("dT/dt  (°C/s)")
    ax.set_title(f"Temperature Rate-of-Change (smoothed, window={SMOOTH})")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "13_temperature_rate_of_change.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14 — OBC_PWR_V with Z-Score Anomaly Overlay
# ─────────────────────────────────────────────────────────────────────────────
def plot_voltage_detail(cdh: pd.DataFrame, output_dir: str) -> str:
    """OBC_PWR_V and SYS_V time series — fine-scale voltage view."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, col, title in [
        (axes[0], "OBC_PWR_V", "OBC Power Voltage"),
        (axes[1], "SYS_V",     "System Bus Voltage"),
    ]:
        if not _plot_series_with_band(ax, cdh, col, col, "steelblue", linewidth=1.6):
            continue
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[1].set_xlabel("Elapsed Time (s)")
    plt.tight_layout()
    return _save(fig, output_dir, "14_voltage_detail.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15 — Temperature Pairplot
# ─────────────────────────────────────────────────────────────────────────────
def plot_temperature_pairplot(cdh: pd.DataFrame, output_dir: str) -> str:
    """
    Seaborn pairplot of the 5 most dynamic temperature channels.
    Diagonal shows KDE; off-diagonal shows scatter.
    """
    key_temps = ["TEMP_OBDH", "TEMP_COMMS", "TEMP_SOLAR",
                 "TEMP_EPS", "TEMP_BACKPLANE"]
    cols  = [c for c in key_temps if c in cdh.columns]
    sub   = cdh[cols].rename(columns=lambda c: c.replace("TEMP_", ""))

    pg = sns.pairplot(sub, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12},
                      diag_kws={"fill": True})
    pg.figure.suptitle("Temperature Channel Pairplot — Key Sensors", y=1.02)
    path = os.path.join(output_dir, "15_temperature_pairplot.png")
    pg.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(pg.figure)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 16 — IMU 3-Axis Sensor Time-Series (Accel / Gyro / Magnetometer)
# ─────────────────────────────────────────────────────────────────────────────
def plot_imu_sensors(adcs: pd.DataFrame, output_dir: str) -> str:
    """
    Three stacked panels — one each for accelerometer, gyroscope, and
    magnetometer axes. Only called when at least one axis is non-zero.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    sensor_groups = [
        ("Accelerometer (g)", ["ACCEL_X", "ACCEL_Y", "ACCEL_Z"], ["X", "Y", "Z"]),
        ("Gyroscope (°/s)",   ["GYRO_X",  "GYRO_Y",  "GYRO_Z"],  ["X", "Y", "Z"]),
        ("Magnetometer (µT)", ["MAG_X",   "MAG_Y",   "MAG_Z"],   ["X", "Y", "Z"]),
    ]

    for ax, (title, cols, labels) in zip(axes, sensor_groups):
        for i, (col, label) in enumerate(zip(cols, labels)):
            _plot_series_with_band(
                ax, adcs, col, label, PALETTE[i % len(PALETTE)], linewidth=1.6
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(ncol=3, fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Elapsed Time (s)")
    fig.suptitle("ADCS IMU Sensor Axes — Full Session", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, output_dir, "16_imu_sensors.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17 — IMU Vector Magnitudes
# ─────────────────────────────────────────────────────────────────────────────
def plot_imu_magnitudes(adcs: pd.DataFrame, output_dir: str) -> str:
    """
    Euclidean magnitude of each IMU vector over time — highlights overall
    motion intensity without needing to parse individual axes.
    """
    fig, ax = plt.subplots(figsize=WIDE)

    groups = [
        (["ACCEL_X", "ACCEL_Y", "ACCEL_Z"], "Accel magnitude (g)",    PALETTE[0]),
        (["GYRO_X",  "GYRO_Y",  "GYRO_Z"],  "Gyro magnitude (°/s)",  PALETTE[1]),
        (["MAG_X",   "MAG_Y",   "MAG_Z"],   "Mag magnitude (µT)",    PALETTE[2]),
    ]
    for cols, label, color in groups:
        available = [c for c in cols if c in adcs.columns]
        if len(available) == 3:
            mag = np.sqrt(sum(adcs[c] ** 2 for c in available))
            ax.plot(adcs["elapsed_s"], mag, label=label,
                    linewidth=1.8, color=color)

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Magnitude")
    ax.set_title("IMU Vector Magnitudes — Accelerometer / Gyroscope / Magnetometer")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "17_imu_magnitudes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 18 — Reaction Wheel Speed
# ─────────────────────────────────────────────────────────────────────────────
def plot_wheel_speed(adcs: pd.DataFrame, output_dir: str) -> str:
    """
    Reaction wheel speed (signed RPM counts) over the session.
    Negative values indicate reverse spin direction.
    """
    fig, ax = plt.subplots(figsize=WIDE)
    _plot_series_with_band(ax, adcs, "WHEEL_SPEED", "Wheel Speed", PALETTE[3], linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5,
               label="Zero (stationary)")
    ax.fill_between(adcs["elapsed_s"], adcs["WHEEL_SPEED"], 0,
                    where=adcs["WHEEL_SPEED"] > 0, alpha=0.15, color=PALETTE[3],
                    label="Forward spin")
    ax.fill_between(adcs["elapsed_s"], adcs["WHEEL_SPEED"], 0,
                    where=adcs["WHEEL_SPEED"] < 0, alpha=0.15, color="crimson",
                    label="Reverse spin")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Wheel Speed (signed counts)")
    ax.set_title("Reaction Wheel Speed — Full Session")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "18_wheel_speed.png")


# ─────────────────────────────────────────────────────────────────────────────
# 19 — Sun Sensor Readings and Sun Angle
# ─────────────────────────────────────────────────────────────────────────────
def plot_sun_sensors(adcs: pd.DataFrame, output_dir: str) -> str:
    """All sun sensor channels plus derived sun angle."""
    sun_cols = [c for c in adcs.columns if c.startswith("SUN_SENSOR_")]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    for i, col in enumerate(sun_cols):
        _plot_series_with_band(
            ax, adcs, col, col.replace("SUN_SENSOR_", "Face "),
            PALETTE[i % len(PALETTE)], linewidth=1.6
        )
    ax.set_ylabel("ADC Count")
    ax.set_title("Sun Sensor Readings per Face")
    ax.legend(ncol=4, fontsize=8, loc="upper right")

    ax2 = axes[1]
    if "SUN_ANGLE" in adcs.columns:
        _plot_series_with_band(
            ax2, adcs, "SUN_ANGLE", "Sun Angle", PALETTE[8 % len(PALETTE)], linewidth=1.8
        )
        ax2.set_ylabel("Sun Angle (counts)")
        ax2.set_title("Derived Sun Angle")
        ax2.legend(fontsize=9)

    axes[-1].set_xlabel("Elapsed Time (s)")
    fig.suptitle("Sun Sensor Data — Full Session", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, output_dir, "19_sun_sensors.png")


# ─────────────────────────────────────────────────────────────────────────────
# 20 — Magnetorquer Commands (X / Y / Z)
# ─────────────────────────────────────────────────────────────────────────────
def plot_magnetorquer(adcs: pd.DataFrame, output_dir: str) -> str:
    """Magnetorquer command values on each axis."""
    cols   = ["MAGNETORQUER_X", "MAGNETORQUER_Y", "MAGNETORQUER_Z"]
    labels = ["X", "Y", "Z"]

    fig, ax = plt.subplots(figsize=WIDE)
    for col, label, color in zip(cols, labels, PALETTE):
        _plot_series_with_band(ax, adcs, col, f"MTQ {label}", color, linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.4)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Command Value (signed counts)")
    ax.set_title("Magnetorquer Commands — X / Y / Z Axes")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "20_magnetorquer.png")


def plot_comms_link_health(cdh: pd.DataFrame, output_dir: str) -> str:
    """Baseline comms channels with run-variation bands when available."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    if not _plot_series_with_band(ax, cdh, "RSSI", "RSSI", PALETTE[6 % len(PALETTE)], linewidth=1.8):
        ax.text(0.5, 0.5, "RSSI unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("RSSI")
    ax.set_title("Comms Link Strength (RSSI)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if not _plot_series_with_band(ax2, cdh, "FRAME_ERR", "FRAME_ERR", PALETTE[1], linewidth=1.8):
        ax2.text(0.5, 0.5, "FRAME_ERR unavailable", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_ylabel("Frame Error Count")
    ax2.set_title("Comms Frame Errors")
    ax2.legend(fontsize=9)
    ax2.set_xlabel("Elapsed Time (s)")

    plt.tight_layout()
    return _save(fig, output_dir, "21_comms_link_health.png")


def plot_comms_anomaly_view(cdh: pd.DataFrame, output_dir: str) -> str:
    """Anomaly-focused comms view with low-RSSI and high-error regions highlighted."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    if "RSSI" in cdh.columns and "elapsed_s" in cdh.columns:
        rssi = pd.to_numeric(cdh["RSSI"], errors="coerce")
        thr = float(rssi.quantile(0.05)) if rssi.notna().any() else np.nan
        _plot_series_with_band(ax, cdh, "RSSI", "RSSI", PALETTE[6 % len(PALETTE)], linewidth=1.8)
        if not np.isnan(thr):
            ax.axhline(thr, color="crimson", linestyle="--", linewidth=1.3,
                       label=f"Low RSSI threshold ({thr:.2f})")
    else:
        ax.text(0.5, 0.5, "RSSI unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("RSSI")
    ax.set_title("Comms Anomaly Detection View — RSSI Drops")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if "FRAME_ERR" in cdh.columns and "elapsed_s" in cdh.columns:
        ferr = pd.to_numeric(cdh["FRAME_ERR"], errors="coerce")
        thr2 = float(ferr.quantile(0.95)) if ferr.notna().any() else np.nan
        _plot_series_with_band(ax2, cdh, "FRAME_ERR", "FRAME_ERR", PALETTE[1], linewidth=1.8)
        if not np.isnan(thr2):
            ax2.axhline(thr2, color="crimson", linestyle="--", linewidth=1.3,
                        label=f"Burst threshold ({thr2:.2f})")
    else:
        ax2.text(0.5, 0.5, "FRAME_ERR unavailable", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_ylabel("Frame Error Count")
    ax2.set_title("Comms Anomaly Detection View — Error Bursts")
    ax2.legend(fontsize=9)
    ax2.set_xlabel("Elapsed Time (s)")

    plt.tight_layout()
    return _save(fig, output_dir, "22_comms_anomaly_view.png")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all plots
# ─────────────────────────────────────────────────────────────────────────────
def run_all_plots(cdh: pd.DataFrame, adcs: pd.DataFrame, output_dir: str) -> list[str]:
    """Run every plot function and return the list of saved file paths."""
    anom_dir = os.path.join(output_dir, "anomaly_detection")
    os.makedirs(anom_dir, exist_ok=True)

    paths = []

    steps = [
        ("01 Board temperatures",         lambda: plot_board_temperatures(cdh, output_dir)),
        ("02 OBDH ramp",                  lambda: plot_obdh_ramp(cdh, anom_dir)),
        ("03 Power rails",                lambda: plot_power_rails(cdh, output_dir)),
        ("04 Thermistor anomaly",         lambda: plot_thermistor_anomaly(cdh, anom_dir)),
        ("05 Temperature correlation",    lambda: plot_temperature_correlation(cdh, anom_dir)),
        ("06 Residual rail voltages",     lambda: plot_residual_rails(cdh, anom_dir)),
        ("07 Summary dashboard",          lambda: plot_summary_dashboard(cdh, output_dir)),
        ("08 Temperature distributions",  lambda: plot_temperature_distributions(cdh, output_dir)),
        ("09 Current draw",               lambda: plot_current_draw(cdh, output_dir)),
        ("10 Temperature deltas",         lambda: plot_temperature_deltas(cdh, output_dir)),
        ("11 OBDH rolling stats",         lambda: plot_obdh_rolling_stats(cdh, anom_dir)),
        ("12 Packet timing jitter",       lambda: plot_packet_timing(cdh, anom_dir)),
        ("13 Temperature rate-of-change", lambda: plot_temperature_roc(cdh, anom_dir)),
        ("14 Voltage detail",             lambda: plot_voltage_detail(cdh, anom_dir)),
        ("22 Comms anomaly view",         lambda: plot_comms_anomaly_view(cdh, anom_dir)),
        ("15 Temperature pairplot",       lambda: plot_temperature_pairplot(cdh, output_dir)),
        ("21 Comms link health",          lambda: plot_comms_link_health(cdh, output_dir)),
    ]

    for label, fn in steps:
        print(f"  Plotting {label} …")
        paths.append(fn())

    # ── ADCS-specific plots ────────────────────────────────────────────────────
    imu_cols = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z", "GYRO_X", "GYRO_Y", "GYRO_Z",
                "MAG_X", "MAG_Y", "MAG_Z"]
    has_imu = (
        not adcs.empty
        and any(c in adcs.columns for c in imu_cols)
        and any(adcs[c].abs().max() > 0 for c in imu_cols if c in adcs.columns)
    )
    if has_imu:
        print("  Plotting 16 IMU sensors …")
        paths.append(plot_imu_sensors(adcs, output_dir))
        print("  Plotting 17 IMU magnitudes …")
        paths.append(plot_imu_magnitudes(adcs, output_dir))

    if (not adcs.empty and "WHEEL_SPEED" in adcs.columns
            and adcs["WHEEL_SPEED"].dropna().abs().max() > 0):
        print("  Plotting 18 Wheel speed …")
        paths.append(plot_wheel_speed(adcs, output_dir))

    sun_cols = [c for c in adcs.columns if c.startswith("SUN_SENSOR_")] if not adcs.empty else []
    if sun_cols and any(adcs[c].abs().max() > 0 for c in sun_cols):
        print("  Plotting 19 Sun sensors …")
        paths.append(plot_sun_sensors(adcs, output_dir))

    mtq_cols = ["MAGNETORQUER_X", "MAGNETORQUER_Y", "MAGNETORQUER_Z"]
    if (not adcs.empty and any(c in adcs.columns for c in mtq_cols)
            and any(adcs[c].abs().max() > 0 for c in mtq_cols if c in adcs.columns)):
        print("  Plotting 20 Magnetorquer …")
        paths.append(plot_magnetorquer(adcs, output_dir))

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# E2 — ML Metrics Bar Chart (Accuracy / FPR / F1 / AUC-ROC)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ml_metrics_bars(ml_results: list[dict], output_dir: str) -> str:
    """
    Grouped bar chart comparing Accuracy, FPR, F1, and AUC-ROC across
    ZScore / IsolationForest / Autoencoder, averaged over all variants.
    """
    detectors = ["ZScore", "RobustRollingZScore", "IsolationForest", "Autoencoder"]
    metrics   = ["accuracy", "fpr", "f1", "auc_roc"]
    labels_m  = ["Accuracy", "FPR", "F1", "AUC-ROC"]
    DET_COLORS = [PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[2]]

    df = pd.DataFrame(ml_results)
    if df.empty or "detector" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No ML results available", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "E2_ml_metrics_bars.png")

    # Aggregate — mean per detector across all tags/variants
    agg = (df[df["detector"].isin(detectors)]
           .groupby("detector")[metrics]
           .mean()
           .reindex(detectors))

    x   = np.arange(len(metrics))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))

    center = (len(detectors) - 1) / 2.0
    for i, (det, color) in enumerate(zip(detectors, DET_COLORS)):
        vals = agg.loc[det].to_numpy() if det in agg.index else np.full(len(metrics), np.nan)
        bars = ax.bar(x + (i - center) * w, vals, width=w, label=det,
                      color=color, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_m)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5,
               label="0.5 reference")
    ax.set_title("ML Detector Performance — Averaged over All Injection Variants")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "E2_ml_metrics_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
# E3 — Confusion Matrix Grid
# ─────────────────────────────────────────────────────────────────────────────
def plot_ml_confusion_grid(ml_results: list[dict], output_dir: str) -> str:
    """
    3-panel confusion matrix heatmap (one per detector), summed over all
    injection variants.
    """
    detectors  = ["ZScore", "RobustRollingZScore", "IsolationForest", "Autoencoder"]
    DET_COLORS = ["Blues", "Purples", "Oranges", "Greens"]

    df = pd.DataFrame(ml_results)
    if df.empty or "detector" not in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No ML results", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "E3_confusion_matrices.png")

    n_det = len(detectors)
    fig, axes = plt.subplots(1, n_det, figsize=(4.2 * n_det, 4.5))
    if n_det == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — All Injection Variants Summed",
                 fontsize=12, fontweight="bold")

    for ax, det, cmap in zip(axes, detectors, DET_COLORS):
        sub = df[df["detector"] == det] if "detector" in df.columns else pd.DataFrame()
        if sub.empty or "tp" not in sub.columns:
            ax.set_title(det)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        tp = int(sub["tp"].sum())
        tn = int(sub["tn"].sum())
        fp = int(sub["fp"].sum())
        fn = int(sub["fn"].sum())
        cm = np.array([[tn, fp], [fn, tp]])

        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Pred Normal", "Pred Anomaly"],
                    yticklabels=["True Normal", "True Anomaly"],
                    cbar=False, linewidths=0.5)
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        ax.set_title(f"{det}\nAccuracy = {acc:.3f}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, output_dir, "E3_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# E5 — Latency Distribution per Detector
# ─────────────────────────────────────────────────────────────────────────────
def plot_ml_latency_distribution(ml_results: list[dict], output_dir: str) -> str:
    """
    Box plot / strip plot of detection latency (seconds) per detector,
    split by injection severity and detector type.
    """
    detectors = ["ZScore", "RobustRollingZScore", "IsolationForest", "Autoencoder"]
    df = pd.DataFrame(ml_results)
    if df.empty or "latency_s" not in df.columns or "detector" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No latency data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "E5_latency_distribution.png")

    sub = df[df["detector"].isin(detectors) & df["latency_s"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    if sub.empty:
        ax.text(0.5, 0.5, "All anomalies missed (no latency recorded)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
    else:
        sns.boxplot(data=sub, x="detector", y="latency_s", order=detectors,
                    color="lightgray",
                    width=0.4, fliersize=4, ax=ax)
        sns.stripplot(data=sub, x="detector", y="latency_s", order=detectors,
                      color="black", alpha=0.4, size=4, jitter=True, ax=ax)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Detector")
    ax.set_ylabel("Detection Latency (s)")
    ax.set_title("Detection Latency Distribution per Detector")
    plt.tight_layout()
    return _save(fig, output_dir, "E5_latency_distribution.png")


def plot_all_ml_evaluation(ml_results: list[dict], output_dir: str) -> list[str]:
    """Run E2, E3, E5 ML evaluation plots and return saved paths."""
    return [
        plot_ml_metrics_bars(ml_results, output_dir),
        plot_ml_confusion_grid(ml_results, output_dir),
        plot_ml_latency_distribution(ml_results, output_dir),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated — ML metrics grouped by experiment family
# ─────────────────────────────────────────────────────────────────────────────
def plot_ml_metrics_by_family(all_ml: list[dict], output_dir: str) -> str:
    """
    Grouped bar chart of mean ± std for Accuracy / FPR / Recall / F1 / AUC-ROC,
    one subplot per experiment family (AccelerometerTest, GyroTest,
    ReactionWheelTest, ThermalTest, Baseline), with the three detectors as colour groups.
    Averaged across all injection variants within each family.
    """
    detectors  = ["ZScore", "RobustRollingZScore", "IsolationForest", "Autoencoder"]
    metrics    = ["accuracy", "fpr", "recall", "f1", "auc_roc"]
    labels_m   = ["Accuracy", "FPR", "Recall", "F1", "AUC-ROC"]
    families   = ["AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest", "Baseline"]
    DET_COLORS = [PALETTE[0], PALETTE[3], PALETTE[1], PALETTE[2]]

    df = pd.DataFrame(all_ml)
    if df.empty or "detector" not in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No aggregated ML results", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "AGG_ml_metrics_by_family.png")

    # Keep only families that actually appear in the data
    present_families = [f for f in families if f in df.get("family", pd.Series()).values]
    if not present_families:
        present_families = families

    n_fam = len(present_families)
    fig, axes = plt.subplots(1, n_fam, figsize=(5 * n_fam, 6), sharey=True)
    if n_fam == 1:
        axes = [axes]

    fig.suptitle("ML Detector Performance by Experiment Family\n(mean ± std across all injection variants)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(metrics))
    w = 0.18

    for ax, family in zip(axes, present_families):
        sub = df[(df["family"] == family) & df["detector"].isin(detectors)]
        agg_mean = (sub.groupby("detector")[metrics].mean().reindex(detectors)
                    if not sub.empty else pd.DataFrame(index=detectors, columns=metrics))
        agg_std  = (sub.groupby("detector")[metrics].std().reindex(detectors)
                    if not sub.empty else pd.DataFrame(index=detectors, columns=metrics))

        center = (len(detectors) - 1) / 2.0
        for i, (det, color) in enumerate(zip(detectors, DET_COLORS)):
            if det in agg_mean.index:
                vals = agg_mean.loc[det].to_numpy(dtype=float)
                errs = agg_std.loc[det].to_numpy(dtype=float)
            else:
                vals = np.full(len(metrics), np.nan)
                errs = np.full(len(metrics), np.nan)

            bars = ax.bar(x + (i - center) * w, vals, width=w, label=det,
                          color=color, edgecolor="black", linewidth=0.5,
                          yerr=errs, capsize=3, error_kw={"elinewidth": 0.8})
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(family, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_m, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.20)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Score")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DET_COLORS]
    fig.legend(handles, detectors, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return _save(fig, output_dir, "AGG_ml_metrics_by_family.png")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated — IsolationForest feature importances heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_if_feature_importances(all_ml: list[dict], output_dir: str) -> str:
    """
    Heatmap of IsolationForest feature importances averaged across all
    experiments, grouped by injection type (anomaly_tag rows, feature columns).
    Only the top-20 features by mean importance are shown.
    """
    df = pd.DataFrame(all_ml)
    if df.empty or "detector" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No IF results", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "AGG_if_feature_importances.png")

    if_rows = df[df["detector"] == "IsolationForest"].copy()
    if if_rows.empty or "feature_importances" not in if_rows.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No IsolationForest feature importance data",
                ha="center", va="center", fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "AGG_if_feature_importances.png")

    # Expand feature_importances lists into a wide DataFrame
    records = []
    for _, row in if_rows.iterrows():
        imps  = row.get("feature_importances")
        fnames = row.get("feature_names")
        tag = row.get("anomaly_tag") or row.get("tag") or "unknown"
        if not isinstance(imps, list):
            continue
        if not imps:
            continue

        # Backward compatibility: older results may store original feature names
        # while importances are in transformed space (subset/PCA). Reconcile here.
        if not isinstance(fnames, list) or not fnames:
            fnames = [f"f{i}" for i in range(len(imps))]
        if len(imps) != len(fnames):
            n = min(len(imps), len(fnames))
            if n == 0:
                continue
            imps = imps[:n]
            fnames = fnames[:n]

        rec = {"anomaly_tag": tag}
        rec.update(dict(zip(fnames, imps)))
        records.append(rec)

    if not records:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Feature importance lists are empty or mismatched",
                ha="center", va="center", fontsize=13, transform=ax.transAxes)
        plt.tight_layout()
        return _save(fig, output_dir, "AGG_if_feature_importances.png")

    wide = pd.DataFrame(records).groupby("anomaly_tag").mean()

    # Top-20 features by mean importance across all injection types
    top_features = wide.mean(axis=0).nlargest(20).index.tolist()
    wide = wide[top_features]

    fig, ax = plt.subplots(figsize=(14, max(6, len(wide) * 0.45)))
    sns.heatmap(wide, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                annot=(len(wide) <= 20), fmt=".3f", annot_kws={"size": 7},
                cbar_kws={"label": "Mean Feature Importance"})
    ax.set_title("IsolationForest Feature Importances by Injection Type\n(averaged across all experiments & variants)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Injection Type", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    return _save(fig, output_dir, "AGG_if_feature_importances.png")
