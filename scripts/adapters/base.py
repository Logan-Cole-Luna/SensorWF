"""
base.py -- Abstract base class for SensorWF domain adapters.

Each adapter implements M1 (data ingestion) for a specific domain and
returns a standardised DataFrame compatible with pipeline_core M2/M3.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class DomainAdapter(ABC):
    """
    Abstract M1 adapter: loads raw domain data and returns a clean,
    standardised DataFrame for downstream SensorWF modules.

    Required columns in returned DataFrame:
      timestamp : pd.Timestamp  (datetime index)
      elapsed_s : float          (seconds from session start)
      <channel> : float          (one column per sensor channel)
    """

    name: str = "AbstractDomain"
    description: str = ""
    channels: list[str] = []
    native_hz: float = 1.0

    @abstractmethod
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load raw file(s) → normalised DataFrame."""

    @abstractmethod
    def get_quality_config(self) -> dict:
        """Return quality-assessment config dict for pipeline_core.run_quality_assessment."""

    @abstractmethod
    def get_feature_config(self) -> dict:
        """Return feature-engineering config (window, channels subset, etc.)."""

    @abstractmethod
    def get_ontology_path(self) -> str:
        """Return path to domain OWL ontology file."""

    @abstractmethod
    def get_fault_types(self) -> list[dict]:
        """Return list of fault-type dicts understood by the domain injector."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channels={self.channels})"
