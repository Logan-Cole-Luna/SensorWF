"""
parser.py — SCOTTI archive parser
Reads the Analysis .txt file and returns raw CDH and ADCS DataFrames.
"""

import re
import pandas as pd


_TS_RE    = re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d+) (\d{2}/\d{2}/\d{4})\s+\(([\w ]+)\)')
_FIELD_RE = re.compile(r'^\t\t(\w+)\s+=\s+(.+)$')


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
                try:
                    ts = pd.to_datetime(
                        f"{date_str} {time_str}", format="%d/%m/%Y %H:%M:%S.%f"
                    )
                except Exception:
                    ts = None
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
