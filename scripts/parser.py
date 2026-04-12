"""
parser.py — SCOTTI archive parser
Reads Analysis or Debug .txt files and returns raw CDH and ADCS DataFrames.
Falls back to Debug format when the Analysis file is empty/missing.
"""

import re
import pandas as pd


_TS_RE    = re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d+) (\d{2}/\d{2}/\d{4})\s+\(([\w ]+)\)')
_FIELD_RE = re.compile(r'^\t\t(\w+)\s+=\s+(.+)$')

# Debug file: "DD/MM/YYYY HH:MM:SS CDH Plot <space-sep values>"
_DBG_RE = re.compile(
    r'^(\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}:\d{2}) (CDH|ADCS Sensor) Plot (.+)$'
)

# Ordered field names for CDH Plot line (87 positional values)
_CDH_FIELDS = [
    "FSW_DB_VER","DAY","HOUR","MINUTE","SECOND","SC_ID",
    "CMD_RCV_CNT","CMD_ACC_CNT","CMD_REJ_CNT","CMD_INC_CNT",
    "PAYLOAD_STATUS","PLATFORM_PWR_STATUS","ADCS_PWR_STATUS",
    "PAYLOAD_PWR_STATUS","SPR_PWR_STATUS",
    "SOLAR_V","SOLAR_S","EXT_V","EXT_S",
    "BATT_CHR_V","BATT_CHR_I","BATT_CHR_STATUS",
    "SYS_V","SYS_I","SPR_V","SPR_I",
    "PLAT_5V_V","PLAT_5V_I","ADCS_5V_V","ADCS_5V_I",
    "PAYLOAD_5V_V","PAYLOAD_5V_I","OBC_PWR_V","OBC_PWR_I",
    "ADCS_PROC_PWR_V","ADCS_PROC_PWR_I","ADCS_PER_PWR_V","ADCS_PER_PWR_I",
    "COMMS_PWR_V","COMMS_PWR_I",
    "TEMP_SOLAR","TEMP_EPS","TEMP_BATTERY","TEMP_BACKPLANE","TEMP_OBDH",
    "TEMP_ADCS","TEMP_WHEEL","TEMP_COMMS","TEMP_PAYLOAD",
    "TEMP_EPS_L","TEMP_OBDH_L","TEMP_ADCS_L","TEMP_WHEEL_L",
    "TEMP_COMMS_L","TEMP_PAYLOAD_L",
    "FREQ","RSSI","TX_POWER","FRAME_ERR","TEMP_THERM_EXP_L",
    "THERMISTOR_ROD_1","THERMISTOR_ROD_2","THERMISTOR_ROD_3","THERMISTOR_ROD_4",
    "THERMISTOR_ROD_5","THERMISTOR_ROD_6","THERMISTOR_ROD_7","THERMISTOR_ROD_8",
    "THERMISTOR_ROD_9","THERMISTOR_ROD_10","THERMISTOR_ROD_11","THERMISTOR_ROD_12",
    "THERMISTOR_ROD_13","THERMISTOR_ROD_14","THERMISTOR_ROD_15","THERMISTOR_ROD_16",
    "THERMISTOR_BATH_1","THERMISTOR_BATH_2","THERMISTOR_BATH_3",
    "THERMISTOR_EXP3","THERMISTOR_HEATER","THERMAL_EXP_SW_VERSION",
    "OBC_SW_VERSION","PWR_SW_VERSION","COMMS_SW_VERSION",
    "ADCS_SW_VERSION","ACTUATOR_SW_VERSION",
]

# Ordered field names for ADCS Sensor Plot line (30 positional values)
_ADCS_FIELDS = [
    "ADCS_MODE",
    "ACCEL_X","ACCEL_Y","ACCEL_Z",
    "GYRO_X","GYRO_Y","GYRO_Z",
    "MAG_X","MAG_Y","MAG_Z",
    "SUN_SENSOR_1a","SUN_SENSOR_1b",
    "SUN_SENSOR_2a","SUN_SENSOR_2b",
    "SUN_SENSOR_3a","SUN_SENSOR_3b",
    "SUN_SENSOR_4a","SUN_SENSOR_4b",
    "SUN_ANGLE","WHEEL_SPEED",
    "MAGNETORQUER_X","MAGNETORQUER_Y","MAGNETORQUER_Z",
    "KP_VALUE","KD_VALUE","ADCS_VERSION",
    "ADCS_ACC_CNT","ADCS_REJ_CNT","ADCS_INC_CNT","MAG_DEPLOYMENT_STATUS",
]


def parse_analysis_file(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a SCOTTI Analysis archive into two DataFrames.

    Returns
    -------
    cdh : pd.DataFrame
        One row per CDH telemetry packet.
    adcs : pd.DataFrame
        One row per ADCS Sensor telemetry packet.
    """
    cdh_records: list[dict]  = []
    adcs_records: list[dict] = []
    current_record: dict | None = None
    current_type: str | None    = None

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()

            m = _TS_RE.match(line)
            if m:
                _flush(current_record, current_type, cdh_records, adcs_records)
                time_str, date_str, pkt_type = m.group(1), m.group(2), m.group(3)
                ts = None
                dt_str = f"{date_str} {time_str}"
                for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S.%f"):
                    try:
                        ts = pd.to_datetime(dt_str, format=fmt)
                        break
                    except Exception:
                        pass
                current_record = {"timestamp": ts}
                current_type   = pkt_type.strip()
                continue

            m2 = _FIELD_RE.match(line)
            if m2 and current_record is not None:
                name, raw = m2.group(1), m2.group(2).strip()
                try:
                    current_record[name] = float(raw)
                except ValueError:
                    current_record[name] = raw  # keep hex strings for later conversion

    _flush(current_record, current_type, cdh_records, adcs_records)

    cdh  = pd.DataFrame(cdh_records)
    adcs = pd.DataFrame(adcs_records)

    if not cdh.empty:
        cdh  = cdh.sort_values("timestamp").reset_index(drop=True)
    if not adcs.empty:
        adcs = adcs.sort_values("timestamp").reset_index(drop=True)

    return cdh, adcs


def _flush(record, pkt_type, cdh_list, adcs_list):
    if record is None:
        return
    if pkt_type == "CDH":
        cdh_list.append(record)
    elif pkt_type == "ADCS Sensor":
        adcs_list.append(record)


def parse_debug_file(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a SCOTTI Debug archive (.txt) into CDH and ADCS DataFrames.

    The Debug format uses space-separated "Plot" lines that carry decoded
    telemetry values in a fixed positional order.  Timestamps have
    second-level precision only (no sub-second field).

    Returns
    -------
    cdh, adcs : pd.DataFrame  (same schema as parse_analysis_file)
    """
    cdh_records: list[dict]  = []
    adcs_records: list[dict] = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            m = _DBG_RE.match(line)
            if not m:
                continue

            date_str, time_str, pkt_type, values_str = (
                m.group(1), m.group(2), m.group(3), m.group(4)
            )
            try:
                ts = pd.to_datetime(
                    f"{date_str} {time_str}", format="%m/%d/%Y %H:%M:%S"
                )
            except Exception:
                ts = None

            tokens = values_str.split()

            if pkt_type == "CDH":
                fields = _CDH_FIELDS
                records = cdh_records
            else:  # "ADCS Sensor"
                fields = _ADCS_FIELDS
                records = adcs_records

            record: dict = {"timestamp": ts}
            for name, raw in zip(fields, tokens):
                try:
                    record[name] = float(raw)
                except ValueError:
                    record[name] = raw  # keep hex strings for later conversion
            records.append(record)

    cdh  = pd.DataFrame(cdh_records)
    adcs = pd.DataFrame(adcs_records)

    if not cdh.empty:
        cdh  = cdh.sort_values("timestamp").reset_index(drop=True)
    if not adcs.empty:
        adcs = adcs.sort_values("timestamp").reset_index(drop=True)

    return cdh, adcs
