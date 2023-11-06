# -*- coding: utf-8 -*-
"""Classes for managing the ABP processing config and parameters."""

##### IMPORTS #####
# Standard imports
from __future__ import annotations

import logging
from typing import Optional

# Third party imports
from caf.toolkit import config_base
import pydantic
from pydantic import dataclasses

# Local imports
from land_use.abp_processing import database

##### CONSTANTS #####
LOG = logging.getLogger(__name__)


##### CLASSES #####
class _ABPConfig(config_base.BaseConfig):
    """Parameters for running the Address Base Premium data processing."""

    output_folder: pydantic.DirectoryPath  # pylint: disable=no-member
    database_connection_parameters: database.ConnectionParameters


@dataclasses.dataclass
class ShapefileParameters:
    """Parameters for an input shapefile."""

    name: str
    path: pydantic.FilePath  # pylint: disable=no-member
    id_column: str
    crs: str = "EPSG:27700"


class WarehouseConfig(_ABPConfig):
    """Parameters for extracting the LFT warehouse data from ABP."""

    lsoa_shapefile: ShapefileParameters
    year_filter: Optional[int] = None


@dataclasses.dataclass
class FilterCodes:
    """ABP classification codes for filtering."""

    name: str
    voa_scat: list[int] = pydantic.Field(default_factory=list)
    abp_classification: list[str] = pydantic.Field(default_factory=list)

    @pydantic.validator("voa_scat", "abp_classification", pre=True)
    def _csv_to_list(cls, value: list | str) -> list:
        # Pydantic validtor is a class method pylint: disable=no-self-argument
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [i.strip() for i in value.split(",") if i.strip() != ""]
        raise TypeError(f"unexpected type: {type(value)}")


class ABPExtractConfig(_ABPConfig):
    """Parameters for extracting and aggregating data from ABP."""

    filter_codes: FilterCodes
    year: Optional[int] = None
    aggregate_shapefile: Optional[ShapefileParameters] = None


##### FUNCTIONS #####
