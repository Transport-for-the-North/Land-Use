# -*- coding: utf-8 -*-
"""Main module for performing ABP processing and analysis."""

# Built-Ins
from __future__ import annotations

import datetime as dt
import logging
import pathlib
from typing import Optional, Sequence

# Third Party
import geopandas as gpd
import pandas as pd
from psycopg2 import sql
from shapely import geometry

# Local Imports
from land_use.abp_processing import config, database, warehousing

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
def initialise_logger(log_file: pathlib.Path) -> None:
    # TODO(MB) Use CAF.toolkit LogHelper instead of this
    logger = logging.getLogger("land_use.abp_processing")
    logger.setLevel(logging.DEBUG)

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(
        logging.Formatter("[{levelname:^8.8}] {message}", style="{")
    )
    streamhandler.setLevel(logging.INFO)
    logger.addHandler(streamhandler)

    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(
        logging.Formatter("{asctime} [{levelname:^8.8}] {message}", style="{")
    )
    logger.addHandler(filehandler)
    LOG.info("Initialised log file: %s", log_file)


def abp_classification_codes(
    connected_db: database.Database, out_file: pathlib.Path
) -> pd.DataFrame:
    """Fetch ABP classification codes from database and save to CSV."""
    query = """
    SELECT "Concatenated", "Class_Desc", "Primary_Code", "Primary_Desc",
        "Secondary_Code", "Secondary_Desc", "Tertiary_Code",
        "Tertiary_Desc", "Quaternary_Code", "Quaternary_Desc"

    FROM data_common.ab_classification_codes
    ORDER BY "Primary_Desc";
    """
    LOG.info("Fetching ABP classification codes from database")
    data = connected_db.query_to_dataframe(query)
    data.to_csv(out_file, index=False)
    LOG.info("Written: %s", out_file)
    return data


def classification_codes_query(
    voa_scat: Sequence[str],
    abp: Sequence[str],
    year: int | None = None,
) -> sql.Composable:
    """Select data from 'abp_classification' table with filters.

    Parameters
    ----------
    voa_scat: Sequence[str]
        VOA Scat codes for filtering.
    abp: Sequence[str]
        ABP classification code for filtering.
    year : int, optional
        Year to use for filtering data, excludes rows which
        have an end date before this or start date after this.

    Returns
    -------
    sql.Composable
        Select query.
    """
    select_query = """
    SELECT uprn, class_scheme, classification_code,
        start_date, end_date, last_update_date, entry_date
    FROM data_common.abp_classification
    WHERE (
        {filter}
    )
    """
    voa_filter = """
    (class_scheme = 'VOA Special Category' AND classification_code IN ({values}))
    """
    abp_filter = """classification_code IN ({values})"""

    filter_queries = []
    for query, values in ((voa_filter, voa_scat), (abp_filter, abp)):
        if len(values) > 0:
            query = sql.SQL(query.strip()).format(
                values=sql.SQL(",").join(sql.Literal(str(i)) for i in values)
            )
            filter_queries.append(query)

    if len(filter_queries) == 0:
        raise ValueError("no classification filters given")

    sql_query = sql.SQL(select_query.strip()).format(
        filter=sql.SQL(" OR ").join(filter_queries)
    )

    if year is None:
        return sql_query

    date_query_str = [
        "(start_date ISNULL OR DATE_TRUNC('year', start_date) <= {date})",
        "(end_date ISNULL OR DATE_TRUNC('year', end_date) >= {date})",
    ]
    date = sql.Literal(dt.date(year, 1, 1).isoformat())
    date_queries = [sql.SQL(q).format(date=date) for q in date_query_str]

    sql_query = sql.SQL("\n\tAND ").join([sql_query, *date_queries])

    return sql_query


def _positions_geodata(data: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert `data` into GeoDataframe using X / Y coordinate columns."""
    missing = data["x_coordinate"].isna() | data["y_coordinate"].isna()
    if missing.sum() > 0:
        LOG.warning("Missing coordinates for %s rows", missing.sum())

    data.loc[:, "geometry"] = data.apply(
        lambda row: geometry.Point(row["x_coordinate"], row["y_coordinate"]), axis=1
    )
    geodata = gpd.GeoDataFrame(
        data, geometry="geometry", crs=warehousing.CRS_BRITISH_GRID
    )

    for column in geodata.select_dtypes(exclude=("number", "geometry")).columns:
        geodata.loc[:, column] = geodata[column].astype(str)

    duplicated = geodata["uprn"].duplicated().sum()
    if duplicated > 0:
        LOG.warning("%s duplicate UPRNs found", duplicated)

    return geodata


def _fetch_positions(
    connected_db: database.Database,
    select_query: sql.Composable,
    out_file: pathlib.Path,
) -> gpd.GeoDataFrame:
    """Select ABP data and join UPRN coordinates.

    Parameters
    ----------
    connected_df : database.Database
        ABP database to run query on.
    select_query : sql.Composable
        Query to select data, must contain a 'uprn'
        column for joining coordinates to.
    out_file : pathlib.Path
        Path to CSV to save outputs to.

    Returns
    -------
    gpd.GeoDataFrame
        Data fetched.
    """
    query = """
    SELECT q.*, blpu.x_coordinate, blpu.y_coordinate,
        blpu.latitude, blpu.longitude, blpu.postcode_locator

    FROM (
    {query}
    ) q

    LEFT JOIN data_common.abp_blpu blpu ON q.uprn = blpu.uprn
    """
    LOG.info("Extracting coordinates to %s", out_file.name)
    data = connected_db.query_to_dataframe(sql.SQL(query).format(query=select_query))
    data.to_csv(out_file, index=False)
    LOG.info("Written: %s", out_file)
    return _positions_geodata(data)


def _total_by_code(data: pd.DataFrame, zone_column: str) -> pd.DataFrame:
    """Aggregate separate classification area columns.

    These won't necessarily sum to total area due to UPRNs possibly
    having multiple classification codes.
    """
    LOG.info(
        "Total floorspace area may not equal the sum of the floorspace split"
        " by classification code because a single property can have multiple"
        " classification codes. The floorspace from a single property can be"
        " included in multiple code columns but will only be counted once"
        " when calculating the total floorspace by zone."
    )
    code_columns = ["class_scheme", "classification_code"]

    duplicated = data.duplicated(subset=[*code_columns, "uprn"]).sum()
    if duplicated > 0:
        LOG.warning(
            "%s duplicate UPRNs found with the same classification code, only "
            "keeping the first value when aggregating to area by classification code",
            duplicated,
        )

    by_codes = (
        data.drop_duplicates(subset=[*code_columns, "uprn"])
        .groupby([zone_column, *code_columns])["area"]
        .sum()
    )
    by_codes = by_codes.unstack(code_columns).fillna(0)
    by_codes = by_codes.rename(
        columns={
            "VOA Special Category": "VOA_SCat",
            "AddressBase Premium Classification Scheme": "ABP",
        }
    )
    by_codes.columns = pd.Index([f"{i}_{j}" for i, j in by_codes.columns])

    return by_codes


def _aggregate_zoning(
    positions: gpd.GeoDataFrame,
    floorspace: pd.DataFrame,
    zones: gpd.GeoDataFrame,
    id_column: str,
    out_file: pathlib.Path,
) -> None:
    """Add zone ID column to data and aggregate to zone total floorspace."""
    positions: gpd.GeoDataFrame = positions.merge(
        floorspace[["uprn", "area"]],
        on="uprn",
        how="outer",
        indicator="floorspace_merge",
    )
    positions.loc[:, "floorspace_merge"] = positions["floorspace_merge"].replace(
        {"left_only": "positions_only", "right_only": "floorspace_only"}
    )

    zones_positions: gpd.GeoDataFrame = gpd.sjoin(
        positions, zones.reset_index(), how="left", op="within"
    )
    zones_positions = zones_positions.drop(columns=["index_right"], errors="ignore")

    for column in zones_positions.select_dtypes("category").columns:
        zones_positions.loc[:, column] = zones_positions[column].astype(str)

    warehousing.to_kepler_geojson(zones_positions, out_file.with_suffix(".geojson"))

    duplicated = zones_positions.duplicated(subset=["uprn"]).sum()
    if duplicated > 0:
        LOG.warning(
            "%s duplicate UPRNs found, only keeping first value"
            " for each when aggregating to total floorspace",
            duplicated,
        )

    zones_floorspace: gpd.GeoDataFrame = (
        zones_positions.drop_duplicates(subset="uprn")[[id_column, "area"]]
        .groupby(id_column, as_index=False)
        .sum()
    )
    zones_floorspace = zones_floorspace.rename(columns={"area": "total_floorspace"})
    zones_floorspace = zones_floorspace.merge(
        zones, how="outer", on=id_column, validate="1:1"
    ).fillna(0)

    floorspace_by_codes = _total_by_code(zones_positions, id_column)
    zones_floorspace = zones_floorspace.merge(
        floorspace_by_codes, on=id_column, validate="1:1", how="left"
    )

    zones_floorspace = gpd.GeoDataFrame(
        zones_floorspace, geometry="geometry", crs=zones.crs
    )
    out_file = out_file.with_name(out_file.stem + "-aggregated.geojson")
    warehousing.to_kepler_geojson(zones_floorspace, out_file)

    zones_floorspace.drop(columns=["geometry"]).to_csv(
        out_file.with_suffix(".csv"), index=False
    )


def extract_data(
    db_params: database.ConnectionParameters,
    output_folder: pathlib.Path,
    filter_codes: config.FilterCodes,
    year: Optional[int] = None,
    zoning_path: Optional[config.ShapefileParameters] = None,
) -> None:
    """Extract property positions and floorspace from ABP database.

    Filter properties based on VOA SCat and ABP classification codes.

    Parameters
    ----------
    db_params : database.ConnectionParameters
        Parameters for connecting to PostGreSQL database.
    output_folder : pathlib.Path
        Folder to save outputs to.
    filter_codes : config.FilterCodes
        Classification codes for filtering.
    year : int, optional
        Filter properties whose date includes given year, if
        not given doesn't filter out anything based on date.
    zoning_path : config.ShapefileParameters, optional
        Parameters for zoning to aggregate floorspace to.

    Raises
    ------
    ValueError
        If shapefile CRS isn't British National Grid.
    """
    if zoning_path is not None:
        zoning = warehousing.load_shapefile(zoning_path)
        if not zoning.crs.equals(warehousing.CRS_BRITISH_GRID):
            raise ValueError(
                f"shapefile CRS is {zoning.crs.name} (EPSG:{zoning.crs.to_epsg()})"
                f" not {warehousing.CRS_BRITISH_GRID}"
            )
        aggregate_message = f" and aggregating to {zoning_path.name}"
    else:
        aggregate_message = ""

    sub_folder = output_folder / f"{filter_codes.name}"
    if year is not None:
        sub_folder = sub_folder.with_name(sub_folder.name + f"_{year}")
    if zoning_path is not None:
        sub_folder = sub_folder.with_name(sub_folder.name + f"_{zoning_path.name}")
    sub_folder.mkdir(exist_ok=True)

    year_message = "all years" if year is None else f"year {year}"
    message = (
        f"Floorspace data for {year_message}{aggregate_message}"
        f" with classification filter:\n{filter_codes}"
    )
    LOG.info(message)

    with open(sub_folder / "README.txt", "wt", encoding="utf-8") as file:
        file.write(message)

    with database.Database(db_params) as conn:
        abp_classification_codes(conn, output_folder / "ABP_classification_codes.csv")

        query = classification_codes_query(
            [str(i) for i in filter_codes.voa_scat],
            filter_codes.abp_classification,
            year,
        )
        positions = _fetch_positions(
            conn, query, sub_folder / "properties_by_classification-positions.csv"
        )

        floorspace = warehousing.get_warehouse_floorspace(
            conn, sub_folder / "property_floorspace.geojson", query
        )

        if zoning is not None:
            assert zoning_path is not None
            _aggregate_zoning(
                positions,
                floorspace,
                zoning,
                zoning_path.id_column,
                sub_folder / f"property_floorspace_{zoning_path.name}.csv",
            )
