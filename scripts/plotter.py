"""
plotter.py — All visualisations for SLL telemetry analysis.

Each function saves one figure to output_dir and returns the file path.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Board Temperature Time-Series
# ─────────────────────────────────────────────────────────────────────────────
def plot_board_temperatures(cdh: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, col in enumerate(TEMP_COLS):
        if col in cdh.columns:
            ax.plot(cdh["elapsed_s"], cdh[col],
                    label=col.replace("TEMP_", ""), linewidth=1.8,
                    color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Board Temperatures — Full Session")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    plt.tight_layout()
    return _save(fig, output_dir, "01_board_temperatures.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — OBDH Thermal Ramp with Linear Fit
# ─────────────────────────────────────────────────────────────────────────────
def plot_obdh_ramp(cdh: pd.DataFrame, obdh_trend: dict, output_dir: str) -> str:
    slope = obdh_trend["_slope_per_s"]
    intercept = obdh_trend["intercept"]
    r2    = obdh_trend["r_squared"]
    rate  = obdh_trend["slope_per_min"]

    t   = cdh["elapsed_s"]
    y   = cdh["TEMP_OBDH"]
    fit = slope * t + intercept

    fig, ax = plt.subplots(figsize=WIDE)
    ax.scatter(t, y, s=12, color=PALETTE[4], alpha=0.6, zorder=3, label="TEMP_OBDH")
    ax.plot(t, fit, "--", color="crimson", linewidth=2,
            label=f"Linear fit: {rate:.3f} °C/min  (R²={r2:.4f})")
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
        if col in cdh.columns:
            ax.plot(cdh["elapsed_s"], cdh[col],
                    label=col, linewidth=1.8, color=PALETTE[i % len(PALETTE)])
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
        ax.plot(cdh["elapsed_s"], cdh[col],
                linewidth=1.5, alpha=0.75, label=col.replace("THERMISTOR_", ""))
    ax.axhline(77.18, color="red", linestyle="--", linewidth=1.4,
               alpha=0.7, label="Stuck value (77.18 °C)")
    ax.set_title("Thermal Rod Sensors (ROD 1–8) — All Stuck")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=2)

    ax2 = axes[1]
    for i, col in enumerate(bath_cols):
        ax2.plot(cdh["elapsed_s"], cdh[col],
                 linewidth=1.8, label=col.replace("THERMISTOR_", ""),
                 color=PALETTE[i])
    ax2.plot(cdh["elapsed_s"], cdh["TEMP_OBDH"],
             linewidth=1.5, linestyle=":", color="black", label="TEMP_OBDH (real)")
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
        if col in cdh.columns:
            ax.plot(cdh["elapsed_s"], cdh[col], label=label,
                    color=color, linewidth=1.8)
            ax.fill_between(cdh["elapsed_s"], 0, cdh[col],
                            alpha=0.15, color=color)
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
def plot_summary_dashboard(
    cdh: pd.DataFrame,
    obdh_trend: dict,
    zscore_counts: dict,
    output_dir: str,
) -> str:
    slope     = obdh_trend["_slope_per_s"]
    intercept = obdh_trend["intercept"]
    rate      = obdh_trend["slope_per_min"]
    r2        = obdh_trend["r_squared"]
    fit_y     = slope * cdh["elapsed_s"] + intercept

    rod_cols = [c for c in cdh.columns if c.startswith("THERMISTOR_ROD_")]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(
        "Satellite Learning Laboratory — Telemetry Dashboard\n"
        "Archive: 2026-03-02  |  Session: 11:24:54 → 11:27:41",
        fontsize=13, fontweight="bold",
    )

    # A — All board temps
    ax = axes[0, 0]
    for i, col in enumerate(TEMP_COLS):
        if col in cdh.columns:
            ax.plot(cdh["elapsed_s"], cdh[col],
                    label=col.replace("TEMP_", ""), linewidth=1.4,
                    color=PALETTE[i % len(PALETTE)])
    ax.set_title("A — Board Temperatures"); ax.set_ylabel("°C")
    ax.legend(fontsize=7, ncol=2)

    # B — OBDH ramp
    ax = axes[0, 1]
    ax.scatter(cdh["elapsed_s"], cdh["TEMP_OBDH"],
               s=10, color=PALETTE[4], alpha=0.7, zorder=3)
    ax.plot(cdh["elapsed_s"], fit_y, "--", color="crimson", linewidth=2,
            label=f"{rate:.3f} °C/min")
    ax.set_title("B — OBDH Thermal Ramp (Anomaly)"); ax.set_ylabel("°C")
    ax.legend(fontsize=9)

    # C — Primary rails
    ax = axes[1, 0]
    for i, col in enumerate(["SYS_V","PLAT_5V_V","OBC_PWR_V","COMMS_PWR_V"]):
        if col in cdh.columns:
            ax.plot(cdh["elapsed_s"], cdh[col],
                    label=col, linewidth=1.6, color=PALETTE[i])
    ax.set_title("C — Primary Power Rails"); ax.set_ylabel("V")
    ax.legend(fontsize=9)

    # D — Residual rails
    ax = axes[1, 1]
    ax.plot(cdh["elapsed_s"], cdh["ADCS_5V_V"],
            label="ADCS 5V (OFF)", color=PALETTE[0], linewidth=1.8)
    ax.plot(cdh["elapsed_s"], cdh["PAYLOAD_5V_V"],
            label="Payload 5V (OFF)", color=PALETTE[1], linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("D — Residual Voltages (Anomaly)"); ax.set_ylabel("V")
    ax.legend(fontsize=9)

    # E — Stuck thermistor vs real
    ax = axes[2, 0]
    if rod_cols:
        ax.plot(cdh["elapsed_s"], cdh[rod_cols[0]],
                color="red", linewidth=2, label=f"{rod_cols[0]} (stuck)")
    ax.plot(cdh["elapsed_s"], cdh["TEMP_OBDH"],
            color="steelblue", linewidth=1.8, label="TEMP_OBDH (real)")
    ax.plot(cdh["elapsed_s"], cdh["TEMP_EPS"],
            color="green", linewidth=1.8, label="TEMP_EPS (real)")
    ax.set_title("E — Stuck Thermistor vs Real Sensors"); ax.set_ylabel("°C")
    ax.legend(fontsize=9)

    # F — Z-score counts
    ax = axes[2, 1]
    cols_p = list(zscore_counts.keys())
    vals_p = list(zscore_counts.values())
    colors = ["crimson" if v > 0 else "steelblue" for v in vals_p]
    bars = ax.bar([c.replace("TEMP_","") for c in cols_p], vals_p,
                  color=colors, edgecolor="black", linewidth=0.7)
    ax.set_title("F — Z-Score Anomaly Counts per Channel")
    ax.set_ylabel("Count"); ax.set_xlabel("Channel")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, vals_p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=9)

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
    """
    Plots the raw ADC current counts for each powered rail.
    BATT_CHR_I is shown separately as it is signed (positive = charging).
    """
    active = [c for c in CURR_COLS if c in cdh.columns and
              c != "BATT_CHR_I" and cdh[c].notna().any() and cdh[c].max() > 0]
    batt   = "BATT_CHR_I"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for i, col in enumerate(active):
        ax.plot(cdh["elapsed_s"], cdh[col],
                label=col.replace("_I", ""), linewidth=1.8,
                color=PALETTE[i % len(PALETTE)])
    ax.set_title("Subsystem Current Draw (ADC counts, proportional to mA)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Raw Current (ADC counts)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if batt in cdh.columns and cdh[batt].notna().any():
        ax2.plot(cdh["elapsed_s"], cdh[batt],
                 color="crimson", linewidth=1.8, label="BATT_CHR_I")
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
    """
    Shows how much each board has drifted from its own t=0 value,
    making the OBDH ramp visually dominant even if absolute values differ.
    """
    cols = [c for c in TEMP_COLS if c in cdh.columns]

    fig, ax = plt.subplots(figsize=WIDE)
    for i, col in enumerate(cols):
        baseline = cdh[col].iloc[0]
        delta    = cdh[col] - baseline
        ax.plot(cdh["elapsed_s"], delta,
                label=col.replace("TEMP_", ""), linewidth=1.8,
                color=PALETTE[i % len(PALETTE)])

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
def plot_voltage_anomaly_overlay(
    cdh: pd.DataFrame, zscore_masks: dict, output_dir: str
) -> str:
    """
    Plot OBC_PWR_V time series with z-score anomaly points highlighted,
    and additionally show SYS_V for reference.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, col, title in [
        (axes[0], "OBC_PWR_V", "OBC Power Voltage (9 z-score anomalies flagged)"),
        (axes[1], "SYS_V",     "System Bus Voltage (3 z-score anomalies flagged)"),
    ]:
        if col not in cdh.columns:
            continue
        ax.plot(cdh["elapsed_s"], cdh[col],
                color="steelblue", linewidth=1.6, label=col)
        if col in zscore_masks:
            mask = zscore_masks[col]
            ax.scatter(
                cdh.loc[mask, "elapsed_s"], cdh.loc[mask, col],
                color="crimson", zorder=5, s=60,
                label=f"Z-score anomaly ({mask.sum()} pts)",
            )
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title)
        ax.legend(fontsize=9)

    axes[1].set_xlabel("Elapsed Time (s)")
    plt.tight_layout()
    return _save(fig, output_dir, "14_voltage_anomaly_overlay.png")


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
# Convenience: run all plots
# ─────────────────────────────────────────────────────────────────────────────
def run_all_plots(
    cdh: pd.DataFrame,
    adcs: pd.DataFrame,
    report: dict,
    output_dir: str,
) -> list[str]:
    """Run every plot function and return the list of saved file paths."""
    paths = []
    obdh  = report["obdh_trend"]
    zc    = report["zscore_counts"]
    zm    = report["zscore_masks"]

    steps = [
        ("01 Board temperatures",          lambda: plot_board_temperatures(cdh, output_dir)),
        ("02 OBDH ramp",                   lambda: plot_obdh_ramp(cdh, obdh, output_dir)),
        ("03 Power rails",                 lambda: plot_power_rails(cdh, output_dir)),
        ("04 Thermistor anomaly",          lambda: plot_thermistor_anomaly(cdh, output_dir)),
        ("05 Temperature correlation",     lambda: plot_temperature_correlation(cdh, output_dir)),
        ("06 Residual rail voltages",      lambda: plot_residual_rails(cdh, output_dir)),
        ("07 Summary dashboard",           lambda: plot_summary_dashboard(cdh, obdh, zc, output_dir)),
        ("08 Temperature distributions",  lambda: plot_temperature_distributions(cdh, output_dir)),
        ("09 Current draw",                lambda: plot_current_draw(cdh, output_dir)),
        ("10 Temperature deltas",          lambda: plot_temperature_deltas(cdh, output_dir)),
        ("11 OBDH rolling stats",          lambda: plot_obdh_rolling_stats(cdh, output_dir)),
        ("12 Packet timing jitter",        lambda: plot_packet_timing(cdh, output_dir)),
        ("13 Temperature rate-of-change",  lambda: plot_temperature_roc(cdh, output_dir)),
        ("14 Voltage anomaly overlay",     lambda: plot_voltage_anomaly_overlay(cdh, zm, output_dir)),
        ("15 Temperature pairplot",        lambda: plot_temperature_pairplot(cdh, output_dir)),
    ]

    for label, fn in steps:
        print(f"  Plotting {label} …")
        paths.append(fn())

    return paths
