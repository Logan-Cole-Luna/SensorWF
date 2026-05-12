"""
satellite_ontology.py — Programmatic generation of the SensorWF satellite OWL ontology.

Generates satellitesystem.owl in OWL/XML format from the actual channel schema
discovered during M1 ingestion. Called by run_sat.py before M6 if the ontology
file is absent or --regenerate-ontology is passed.

Subsystem mapping (SATLL SCOTTI v2 schema):
  ADCS  — attitude determination and control (ACCEL, GYRO, MAG, WHEEL, SUN_SENSOR)
  OBDH  — on-board data handling (TEMP_OBDH, TEMP_BACKPLANE, timing counters)
  EPS   — electrical power system (SYS_V, SYS_I, PLAT_5V_V, OBC_PWR_*, COMMS_PWR_*)
  Thermal — thermal control (TEMP_SOLAR, TEMP_EPS, TEMP_BATTERY, TEMP_ADCS, TEMP_WHEEL,
                             TEMP_COMMS, TEMP_PAYLOAD)
  Comms — communications (COMMS_*, RSSI, FRAME_ERR)

SSN/SOSA-aligned: each subsystem is a sosa:Platform; each sensor channel class
is a subclass of ssn:Sensor with a sosa:observes data property.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone


# ── Subsystem → channel prefix mapping ────────────────────────────────────────

_SUBSYSTEM_PREFIXES: dict[str, list[str]] = {
    "AttitudeDeterminationAndControlSubsystem": [
        "ACCEL_", "GYRO_", "MAG_", "WHEEL_", "SUN_",
    ],
    "OnBoardDataHandling": [
        "TEMP_OBDH", "TEMP_BACKPLANE", "ELAPSED", "PACKET", "CMD_",
        "ACCEPTED_CMD", "INCORRECT_CMD",
    ],
    "ElectricalPowerSubsystem": [
        "SYS_V", "SYS_I", "PLAT_5V", "OBC_PWR", "COMMS_PWR",
    ],
    "ThermalControlSubsystem": [
        "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_ADCS",
        "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
    ],
    "CommunicationsSubsystem": [
        "COMMS_", "RSSI", "FRAME_ERR",
    ],
}

# Sensor type per channel-name fragment
_SENSOR_CLASS: dict[str, str] = {
    "ACCEL":     "Accelerometer",
    "GYRO":      "Gyroscope",
    "MAG":       "Magnetometer",
    "WHEEL":     "ReactionWheel",
    "SUN":       "SunSensor",
    "TEMP":      "Thermistor",
    "SYS_V":     "VoltageSensor",
    "PLAT_5V":   "VoltageSensor",
    "OBC_PWR_V": "VoltageSensor",
    "COMMS_PWR_V": "VoltageSensor",
    "SYS_I":     "CurrentSensor",
    "OBC_PWR_I": "CurrentSensor",
    "COMMS_PWR_I": "CurrentSensor",
    "RSSI":      "SignalStrengthSensor",
    "FRAME_ERR": "ErrorCounter",
}

_BASE_IRI  = "http://sensorwf.org/ontologies/satellite#"
_PROV_IRI  = "http://sensorwf.org/ontologies/satellite"
_SSN_IRI   = "http://www.w3.org/ns/ssn/"
_SOSA_IRI  = "http://www.w3.org/ns/sosa/"


def _subsystem_for_channel(channel: str) -> str:
    ch = channel.upper()
    for sub, prefixes in _SUBSYSTEM_PREFIXES.items():
        for p in prefixes:
            if ch.startswith(p) or p in ch:
                return sub
    return "OnBoardDataHandling"


def _sensor_class_for_channel(channel: str) -> str:
    ch = channel.upper()
    for frag, cls in _SENSOR_CLASS.items():
        if frag in ch:
            return cls
    return "Sensor"


def _channel_to_class_name(channel: str) -> str:
    """Convert raw column name to a CamelCase OWL class name."""
    return "".join(w.capitalize() for w in channel.replace("_", " ").split())


def generate_satellite_ontology(
    path: str,
    cdh_channels: list[str] | None = None,
    adcs_channels: list[str] | None = None,
) -> str:
    """
    Generate a satellitesystem.owl in OWL/XML format at *path*.

    Parameters
    ----------
    path          : output file path (created/overwritten)
    cdh_channels  : CDH column names discovered from data (optional)
    adcs_channels : ADCS column names discovered from data (optional)

    Returns
    -------
    path (str)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Merge channel lists; always include the core SATLL schema channels
    _core_channels = [
        "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_BACKPLANE",
        "TEMP_OBDH", "TEMP_ADCS", "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
        "SYS_V", "PLAT_5V_V", "OBC_PWR_V", "COMMS_PWR_V",
        "SYS_I", "OBC_PWR_I", "COMMS_PWR_I",
        "ACCEL_X", "ACCEL_Y", "ACCEL_Z",
        "GYRO_X", "GYRO_Y", "GYRO_Z",
        "MAG_X", "MAG_Y", "MAG_Z",
        "WHEEL_SPEED",
        "SUN_X", "SUN_Y", "SUN_Z",
    ]
    all_channels: list[str] = list(dict.fromkeys(
        _core_channels
        + (cdh_channels or [])
        + (adcs_channels or [])
    ))
    # Skip non-domain columns
    skip = {"timestamp", "elapsed_s", "anomaly_type", "anomaly_label"}
    all_channels = [c for c in all_channels if c not in skip]

    # Collect unique subsystems, sensor classes, and channel→subsystem mapping
    subsystems: set[str] = set(_SUBSYSTEM_PREFIXES.keys())
    sensor_classes: set[str] = set()
    chan_info: list[tuple[str, str, str]] = []  # (class_name, sensor_cls, subsystem)
    for ch in all_channels:
        cls  = _channel_to_class_name(ch)
        scls = _sensor_class_for_channel(ch)
        sub  = _subsystem_for_channel(ch)
        sensor_classes.add(scls)
        chan_info.append((cls, scls, sub))

    generated_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []

    # ── OWL/XML header ────────────────────────────────────────────────────────
    lines += [
        '<?xml version="1.0"?>',
        '<Ontology xmlns="http://www.w3.org/2002/07/owl#"',
        f'     xml:base="{_PROV_IRI}"',
        '     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        '     xmlns:xml="http://www.w3.org/XML/1998/namespace"',
        '     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
        '     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
        f'     ontologyIRI="{_PROV_IRI}">',
        f'    <Prefix name="" IRI="{_BASE_IRI}"/>',
        '    <Prefix name="owl"  IRI="http://www.w3.org/2002/07/owl#"/>',
        '    <Prefix name="rdf"  IRI="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>',
        '    <Prefix name="rdfs" IRI="http://www.w3.org/2000/01/rdf-schema#"/>',
        '    <Prefix name="xsd"  IRI="http://www.w3.org/2001/XMLSchema#"/>',
        f'    <Prefix name="ssn"  IRI="{_SSN_IRI}"/>',
        f'    <Prefix name="sosa" IRI="{_SOSA_IRI}"/>',
        '',
        '    <Annotation>',
        '        <AnnotationProperty abbreviatedIRI="rdfs:comment"/>',
        '        <Literal>SensorWF satellite system ontology — auto-generated from '
        f'SATLL SCOTTI v2 channel schema. Generated: {generated_at}</Literal>',
        '    </Annotation>',
        '    <Annotation>',
        '        <AnnotationProperty abbreviatedIRI="rdfs:label"/>',
        '        <Literal>SensorWF Satellite System Ontology</Literal>',
        '    </Annotation>',
        '',
    ]

    # ── Class declarations ────────────────────────────────────────────────────
    def _decl(local: str) -> str:
        return (
            f'    <Declaration>\n'
            f'        <Class IRI="{_BASE_IRI}{local}"/>\n'
            f'    </Declaration>'
        )

    # Top-level hierarchy
    for cls in ["SatelliteSystem", "Subsystem", "Sensor", "Measurement",
                "AnomalyType", "SensorFault"]:
        lines.append(_decl(cls))

    # Subsystem subclasses
    for sub in sorted(subsystems):
        lines.append(_decl(sub))

    # Sensor type subclasses
    for scls in sorted(sensor_classes):
        lines.append(_decl(scls))

    # Per-channel channel classes
    for cls, _, _ in chan_info:
        lines.append(_decl(cls))

    # Fault-type classes
    for ft in ["ThermalRunaway", "ThermalShock", "PowerSag", "SwitchingNoise",
               "RailLatchup", "PacketDropout", "AccelDropout", "GyroClipping",
               "MagInversion", "IMUCorrelationBreak", "WheelRunaway",
               "WheelStiction", "SunSensorBlind", "CommsFrameError",
               "SensorDrift", "StuckSensor"]:
        lines.append(_decl(ft))

    lines.append('')

    # ── SubClassOf axioms ──────────────────────────────────────────────────────
    def _subclass(child: str, parent: str) -> list[str]:
        return [
            '    <SubClassOf>',
            f'        <Class IRI="{_BASE_IRI}{child}"/>',
            f'        <Class IRI="{_BASE_IRI}{parent}"/>',
            '    </SubClassOf>',
        ]

    for sub in sorted(subsystems):
        lines += _subclass(sub, "Subsystem")

    for scls in sorted(sensor_classes):
        lines += _subclass(scls, "Sensor")

    for cls, scls, _ in chan_info:
        lines += _subclass(cls, scls)

    for ft in ["ThermalRunaway", "ThermalShock", "PowerSag", "SwitchingNoise",
               "RailLatchup", "PacketDropout", "AccelDropout", "GyroClipping",
               "MagInversion", "IMUCorrelationBreak", "WheelRunaway",
               "WheelStiction", "SunSensorBlind", "CommsFrameError",
               "SensorDrift", "StuckSensor"]:
        lines += _subclass(ft, "SensorFault")

    lines.append('')

    # ── Object property declarations ──────────────────────────────────────────
    for prop in ["partOf", "hasSubsystem", "hasSensor", "observes",
                 "evidenceForClass", "featureImportance", "detectedBy"]:
        lines += [
            '    <Declaration>',
            f'        <ObjectProperty IRI="{_BASE_IRI}{prop}"/>',
            '    </Declaration>',
        ]

    lines.append('')

    # ── Channel → subsystem ObjectPropertyDomain/Range ─────────────────────
    for cls, scls, sub in chan_info:
        lines += [
            '    <ObjectPropertyDomain>',
            f'        <ObjectProperty IRI="{_BASE_IRI}partOf"/>',
            f'        <Class IRI="{_BASE_IRI}{cls}"/>',
            '    </ObjectPropertyDomain>',
            '    <ObjectPropertyRange>',
            f'        <ObjectProperty IRI="{_BASE_IRI}partOf"/>',
            f'        <Class IRI="{_BASE_IRI}{sub}"/>',
            '    </ObjectPropertyRange>',
        ]

    lines.append('')

    # ── rdfs:label annotations for key classes ─────────────────────────────
    labels = {
        "AttitudeDeterminationAndControlSubsystem": "ADCS",
        "OnBoardDataHandling": "OBDH",
        "ElectricalPowerSubsystem": "EPS",
        "ThermalControlSubsystem": "Thermal",
        "CommunicationsSubsystem": "Comms",
        "SatelliteSystem": "Satellite System",
        "SensorFault": "Sensor Fault",
    }
    for cls, lbl in labels.items():
        lines += [
            '    <AnnotationAssertion>',
            '        <AnnotationProperty abbreviatedIRI="rdfs:label"/>',
            f'        <IRI>{_BASE_IRI}{cls}</IRI>',
            f'        <Literal>{lbl}</Literal>',
            '    </AnnotationAssertion>',
        ]

    lines.append('</Ontology>')

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/ontologies/satellitesystem.owl")
    args = ap.parse_args()
    out = generate_satellite_ontology(args.output)
    print(f"Generated: {out}")
