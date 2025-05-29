# %%
from pathlib import Path
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
import os.path
import logging

import land_use.preprocessing as pp

from land_use import constants, data_processing
from land_use import logging as lu_logging

from caf.base.segmentation import SegmentsSuper
from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

# Parameters
BASE_YEAR = 2023
# Include the base year in here as we are pivoting from it
CROSSOVER_YEAR = 2043
FORECAST_YEARS = [2023, 2033, 2038, 2043, 2048, 2053]
# Location of base population folder which we are pivoting from
BASE_POP_DV = Path(
    r"F:\Working\Land-Use\BASE_POPULATION_WITH_AGE_NTEM\based_on_241220_Populationv2"
)
POPULATION_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs")
HOUSEHOLDS_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\hh_projs")
LMS_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\Labour Market and Skills")
OBR_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\OBR")

IPF_TARGET_OUT_DIR = Path(r"F:\Working\Land-Use\POPULATION_TARGETS\based_on_241220_Populationv2")

ENGLAND_CODE = "E92000001"

# %%
# mapping the various household compositions to the children segmentation
# 1: no children
# 2: 1 or more children
CHILDREN_MAPPING = {
    "One person households: Male": 1,
    "One person households: Female": 1,
    "Households with one dependent child": 2,
    "Households with two dependent children": 2,
    "Households with three or more dependent children": 2,
    "Other households with two or more adults": 1,
    "1 adult female": 1,
    "1 adult male": 1,
    "2 adults": 1,
    "1 adult, 1 child": 2,
    "1 adult, 2+ children": 2,
    "2+ adult 1+ children": 2,
    "3+ person all adult": 1,
    "1 person": 1,
    "2 person (No children)": 1,
    "2 person (1 adult, 1 child)": 2,
    "3 person (No children)": 1,
    "3 person (2 adults, 1 child)": 2,
    "3 person (1 adult, 2 children)": 2,
    "4 person (No children)": 1,
    "4 person (2+ adults, 1+ children)": 2,
    "4 person (1 adult, 3 children)": 2,
    "5+ person (No children)": 1,
    "5+ person (2+ adults, 1+ children)": 2,
    "5+ person (1 adult, 4+ children)": 2,
}

# Set up logging of key inputs as this helps audit trail
LOGGER = logging.getLogger(__name__)
LOGGER = lu_logging.configure_logger(
    output_dir=IPF_TARGET_OUT_DIR, log_name="reformat_population_generation_log"
)

LOGGER.info(f"Process run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
LOGGER.info(f"{BASE_YEAR=}")
LOGGER.info(f"{FORECAST_YEARS=}")
LOGGER.info(f"{BASE_POP_DV=}")
LOGGER.info(f"{POPULATION_DIR=}")
LOGGER.info(f"{HOUSEHOLDS_DIR=}")
LOGGER.info(f"{IPF_TARGET_OUT_DIR=}")


# %%
def main():
    """
    Main function which generates the forecast targets
    Calls various functions which can be included/excluded depending on what needs generating
    """
    IPF_TARGET_OUT_DIR.mkdir(exist_ok=True)

    # PROCESSES FOR THE GROWTH FACTORS
    for rgn in constants.GORS + ["Scotland"]:
        process_base_to_ntem_age(rgn)
    for forecast_year in range(BASE_YEAR, 2054):
        write_ons_pop_growth_factors_from_base(
            base_year=BASE_YEAR, forecast_year=forecast_year
        )
    process_and_save_hh_projections_children()
    process_and_save_projections_1_adult_hhs()
    process_and_save_hh_projections()

    # PROCESSES FOR THE TARGETS
    # creating population targets
    calc_and_output_pop_targets(
        base_dv_path=BASE_POP_DV,
        age_g_factors=POPULATION_DIR / r"preprocessing\pop_growth_factors_from_2023.hdf",
        soc_change=LMS_INPUT_DIR / r"LMS_SOC\preprocessing\soc_pp_change_from_2023.hdf",
        g_adults_growth_factors=HOUSEHOLDS_DIR / r"preprocessing\hh_1_adult_by_g_from_2023_factors.hdf",
        base_year=BASE_YEAR,
        forecast_years=FORECAST_YEARS,
        path_out=IPF_TARGET_OUT_DIR
    )

    # creating household targets
    calc_and_output_hh_targets(
        base_dv_path=Path(
            r"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs"),
        hh_totals_factors=HOUSEHOLDS_DIR / r"preprocessing\hh_totals_from_2023_factors.hdf",
        hh_children_factors=HOUSEHOLDS_DIR / r"preprocessing\hh_children_from_2023_factors.hdf",
        hh_single_adult_no_g_factors=HOUSEHOLDS_DIR / r"preprocessing\hh_1_adult_by_no_g_from_2023_factors.hdf",
        base_year=BASE_YEAR,
        forecast_years=FORECAST_YEARS,
        path_out=IPF_TARGET_OUT_DIR
    )

    # Separate steps - the following are calculated based on the forecast population outputs
    # creating ns-sec household targets (based on the outputs from forecast population)
    calc_and_output_nssec_hh_targets(
        base_pop_dv_path=BASE_POP_DV,
        base_hhs_dv_path=Path(r"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs"),
        forecast_dv_path=Path(
            r"F:\Working\Land-Use\forecast_population_20250515\02_Final Outputs"),
        forecast_years=[2033, 2038, 2043, 2048, 2053],
        path_out=IPF_TARGET_OUT_DIR
    )

    # create household occupancy targets
    calc_and_output_hh_occupancy_targets(
        forecast_dv_path=Path(
            r"F:\Working\Land-Use\forecast_population_20250515\02_Final Outputs"),
        forecast_years=[2033, 2038, 2043, 2048, 2053],
        path_out=IPF_TARGET_OUT_DIR
    )


def write_ons_pop_growth_factors_from_base(base_year: int, forecast_year: int) -> None:

    if base_year < CROSSOVER_YEAR:
        ons_pop_base = fetch_ons_pop_forecasts_up_to_crossover(year=base_year)
    else:
        ons_pop_base = fetch_ons_pop_forecasts_post_crossover(year=base_year)

    if forecast_year < CROSSOVER_YEAR:
        ons_pop_fy = fetch_ons_pop_forecasts_up_to_crossover(year=forecast_year)
    else:
        ons_pop_fy = fetch_ons_pop_forecasts_post_crossover(year=forecast_year)

    pop_growth_factor = ons_pop_fy / ons_pop_base

    filestem = f"pop_growth_factors_from_{base_year}"

    pp.save_preprocessed_hdf(
        source_file_path=POPULATION_DIR / f"{filestem}.hdf",
        df=pop_growth_factor,
        key=f"factors_from_{base_year}_to_{forecast_year}",
        mode="a",
    )


def calc_and_output_pop_targets(
        base_dv_path: Path,
        age_g_factors: Path,
        soc_change: Path,
        g_adults_growth_factors: Path,
        base_year: int,
        forecast_years: list,
        path_out: Path
):
    """
    Calculate the population targets and writes to a hdf (if provided with a path_out)

    Parameters
    ----------
    base_dv_path: Path
        Base population DVectors path
    age_g_factors: Path
        Factors derived from ONS population projections data (can be updated if reading in different targets)
    soc_change: Path
        Proportion split changes derived from LM&S data (can be updated if reading in different targets)
    g_adults_growth_factors: Path
        Factors derived from ONS household projections data (single adult households by gender)
    base_year: int
        Base year of Land Use
    forecast_years: list
        Years to generate forecasts for
    path_out: Path
        Location to output the DVector targets to
    """
    # TODO think about changing the inputs to flow through in one script
    # --------------------------------------------------------
    # Reading in the Base data and the growth factors
    for forecast_year in forecast_years:
        pop_dfs = []
        soc_dfs = []
        pop_g_adults_dfs = []
        # common hdf key across all outputs
        hdf_key = f"targets_{forecast_year}"
        for gor in constants.GORS + ["Scotland"]:
            base_dv = DVector.load(base_dv_path / f"Output P11_{gor}.hdf")

            pop_growth_factor = data_processing.read_dvector_data(
                file_path=age_g_factors,
                geographical_level="RGN2021",
                input_segments=["age_ntem", "g"],
                geography_subset=gor,
                hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
            )

            soc_splits_change = data_processing.read_dvector_data(
                file_path=soc_change,
                geographical_level="RGN2021",
                input_segments=["g", "soc"],
                geography_subset=gor,
                hdf_key=f"change_from_{base_year}_to_{forecast_year}"
            )

            pop_g_adults_growth_factors = data_processing.read_dvector_data(
                file_path=g_adults_growth_factors,
                geographical_level="LAD2019_EWS",
                input_segments=["g", "adults"],
                geography_subset=gor,
                hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
            )

            # --------------------------------------------------------
            # Formatting and calculating targets
            base_pop_age_ntem_g = base_dv.aggregate(segs=["age_ntem", "g", "soc"])

            base_pop_age_ntem_g_gor = base_pop_age_ntem_g.translate_zoning(
                new_zoning=pop_growth_factor.zoning_system,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )

            # -- POPULATION AGE AND GENDER --
            pop_targets = base_pop_age_ntem_g_gor * pop_growth_factor

            base_pop_age_ntem_g_gor = pop_targets.translate_zoning(
                new_zoning=pop_growth_factor.zoning_system,  # fix zoning system to match
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )

            # -- POPULATION SOC (EXCLUDING SOC 4) --
            # also need to have a total for SOC excluding SOC 4
            base_pop_soc = base_pop_age_ntem_g_gor.aggregate(["soc"])

            base_pop_soc_exc_4 = base_pop_soc.filter_segment_value("soc", [1, 2, 3])

            base_pop_soc_exc_4_total = base_pop_soc_exc_4.add_segments(["total"]).aggregate(
                ["total"]
            )

            base_pop_soc_exc_4_total = base_pop_soc_exc_4_total.add_segments(
                ["soc"]
            ).filter_segment_value("soc", [1, 2, 3])

            # -- POPULATION GENDER AND ADULTS --
            base_pop_g_adults = base_dv.aggregate(segs=["g", "adults"]).filter_segment_value(
                "adults", [1]
            )

            base_pop_g_adults_lad19 = base_pop_g_adults.translate_zoning(
                new_zoning=pop_g_adults_growth_factors.zoning_system,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )

            pop_g_adults_targets = base_pop_g_adults_lad19 * pop_g_adults_growth_factors

            # and the lad targets need to be converted to a compatible 2021 zone system,
            # being a combination of lsoa 2021 zones
            pop_g_adults_targets = pop_g_adults_targets.translate_zoning(
                new_zoning=constants.KNOWN_GEOGRAPHIES.get(f"LAD2021-{gor}"),
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )

            # Calculate new SOC splits
            base_pop_gor = base_dv.translate_zoning(pop_growth_factor.zoning_system)

            base_pop_gor = base_pop_gor.aggregate(["g", "soc"]).filter_segment_value(
                "soc", [1, 2, 3]
            )
            base_pop_soc_totals = (
                base_pop_gor.add_segments(["total"])
                .aggregate(["total"])
                .add_segments(["soc"])
                .filter_segment_value("soc", [1, 2, 3])
            )

            base_pop_soc_splits = base_pop_gor / base_pop_soc_totals

            # work out the new targets splits
            soc_target_splits = base_pop_soc_splits + soc_splits_change
            # check for negative splits
            if (soc_target_splits.data < 0).any().any():
                raise ValueError(
                    f"New SOC target splits calculated contain negatives for {forecast_year} - {gor}"
                )

            # the totals here should be the pop_targets without soc 4
            soc_targets = (soc_target_splits * base_pop_soc_exc_4_total).aggregate(["g", "soc"])

            pop_targets_df = pop_targets.data
            soc_targets_df = soc_targets.data
            pop_g_adults_targets_df = pop_g_adults_targets.data

            # Make sure that all dfs are in the same index order (for concatenation)
            pop_targets_df = pop_targets_df.reorder_levels(["age_ntem", "g", "soc"])
            soc_targets_df = soc_targets_df.reorder_levels(["g", "soc"])
            pop_g_adults_targets_df = pop_g_adults_targets_df.reorder_levels(["g", "adults"])

            pop_dfs.append(pop_targets_df)
            soc_dfs.append(soc_targets_df)
            pop_g_adults_dfs.append(pop_g_adults_targets_df)

        # --------------------------------------------------------
        # Outputting
        pop_dfs_output = pd.concat(pop_dfs, axis=1)
        soc_dfs_output = pd.concat(soc_dfs, axis=1)
        pop_g_adults_dfs_output = pd.concat(pop_g_adults_dfs, axis=1)

        path_out = path_out
        message = f"writing to {path_out}, with {hdf_key=}"
        print(message)
        pop_dfs_output.to_hdf(path_out / "pop_targets.hdf", key=hdf_key, mode="a")
        soc_dfs_output.to_hdf(path_out / "soc_targets.hdf", key=hdf_key, mode="a")
        pop_g_adults_dfs_output.to_hdf(path_out / "pop_g_adults_targets.hdf", key=hdf_key, mode="a")

        # return pop_targets, soc_targets, pop_g_adults_targets


def calc_and_output_hh_targets(
        base_dv_path: Path,
        hh_totals_factors: Path,
        hh_children_factors: Path,
        hh_single_adult_no_g_factors: Path,
        base_year: int,
        forecast_years: list,
        path_out: Path
):
    """
    Calculate the household targets and writes to a hdf (if provided with a path_out)

    Parameters
    ----------
    base_dv_path: Path
        Base population DVectors path
    hh_totals_factors: Path
        Factors derived from ONS household projections data (can be updated if reading in different targets)
    hh_children_factors: Path
        Factors derived from ONS household projections data
    hh_single_adult_no_g_factors: Path
        Factors derived from ONS household projections data (single adult households by gender)
    base_year: int
        Base year of Land Use
    forecast_years: list
        Years to generate forecasts for
    path_out: Path
        Location to output the DVector targets to
    """

    # --------------------------------------------------------
    # Reading in the Base data and the growth factors
    for forecast_year in forecast_years:
        children_dfs = []
        totals_dfs = []
        adults_dfs = []
        # common hdf key across all outputs
        hdf_key = f"targets_{forecast_year}"
        for gor in constants.GORS + ["Scotland"]:
            base_dv = DVector.load(base_dv_path / f"Output P13.3_{gor}.hdf")

            totals_growth_factor = data_processing.read_dvector_data(
                file_path=hh_totals_factors,
                geographical_level="RGN2021",
                input_segments=["total"],
                geography_subset=gor,
                hdf_key=f"factors_from_{base_year}_to_{forecast_year}"
            )

            children_growth_factor = data_processing.read_dvector_data(
                file_path=hh_children_factors,
                geographical_level="RGN2021",
                input_segments=["children"],
                geography_subset=gor,
                hdf_key=f"factors_from_{base_year}_to_{forecast_year}"
            )

            single_adults_growth_factors = data_processing.read_dvector_data(
                file_path=hh_single_adult_no_g_factors,
                geographical_level="LAD2019_EWS",
                input_segments=["adults"],
                geography_subset=gor,
                hdf_key=f"factors_from_{base_year}_to_{forecast_year}"
            )

            # --------------------------------------------------------
            # Formatting and calculating targets
            if "total" in base_dv.segmentation.names:
                base_dv_totals = base_dv.aggregate(segs=["total"])
            else:
                base_dv_totals = base_dv.add_segments(new_segs=["total"]).aggregate(
                    segs=["total"]
                )

            # Children targets
            base_dv_children = base_dv.aggregate(segs=["children"])
            base_dv_children_gor = base_dv_children.translate_zoning(
                new_zoning=children_growth_factor.zoning_system,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )
            hh_children_targets = base_dv_children_gor * children_growth_factor

            # Totals targets
            base_dv_totals_gor = base_dv_totals.translate_zoning(
                new_zoning=totals_growth_factor.zoning_system,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )
            hh_totals_targets = base_dv_totals_gor * totals_growth_factor

            # Single adult household targets
            base_dv_single_adults = base_dv.aggregate(segs=["adults"]).filter_segment_value(
                "adults", [1]
                )
            base_dv_single_adults_lad19 = base_dv_single_adults.translate_zoning(
                new_zoning=single_adults_growth_factors.zoning_system,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )
            hh_single_adults_targets = base_dv_single_adults_lad19 * single_adults_growth_factors
            # and the lad targets need to be converted to a compatible 2021 zone system,
            # being a combination of lsoa 2021 zones
            hh_adults_targets = hh_single_adults_targets.translate_zoning(
                new_zoning=constants.KNOWN_GEOGRAPHIES.get(f"LAD2021-{gor}"),
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.NO_WEIGHT,
            )

            children_targets_df = hh_children_targets.data
            totals_targets_df = hh_totals_targets.data
            single_adults_targets_df = hh_adults_targets.data

            children_dfs.append(children_targets_df)
            totals_dfs.append(totals_targets_df)
            adults_dfs.append(single_adults_targets_df)

        # --------------------------------------------------------
        # Outputting
        children_dfs_output = pd.concat(children_dfs, axis=1)
        totals_dfs_output = pd.concat(totals_dfs, axis=1)
        adults_dfs_output = pd.concat(adults_dfs, axis=1)

        message = f"writing to {path_out}, with {hdf_key=}"
        print(message)
        children_dfs_output.to_hdf(path_out / "household_children_targets.hdf", key=hdf_key, mode="a")
        totals_dfs_output.to_hdf(path_out / "household_totals_targets.hdf", key=hdf_key, mode="a")
        adults_dfs_output.to_hdf(path_out / "household_single_adults_targets.hdf", key=hdf_key, mode="a")


def calc_and_output_nssec_hh_targets(
        base_pop_dv_path: Path,
        base_hhs_dv_path: Path,
        forecast_dv_path: Path,
        forecast_years: list,
        path_out: Path
):
    """
    Function to maintain the NS-SEC changes in the household data based on the outcome of the NS-SEC in the population
    Keep separate for now as the NS-SEC household targets are based off the forecast population outputs

    Parameters
    ----------
    base_pop_dv_path: Path
        Base population DVectors path
    base_hhs_dv_path: Path
        Base household DVectors path
    forecast_dv_path: Path
        Forecast population DVectors output
    forecast_years: list
        Years to generate forecasts for
    path_out: Path
        Location to output the DVector targets to
    """

    # Reading in the Base and Forecast population data
    # Also read in the Base household data for later on in the function
    for forecast_year in forecast_years:
        nssec_hh_targets = []
        hdf_key = f"targets_{forecast_year}"
        for gor in constants.GORS + ["Scotland"]:
            print(f"Processing for {forecast_year}, {gor}")
            # Read in base population
            base_pop_dv = DVector.load(base_pop_dv_path / f"Output P11_{gor}.hdf")
            base_pop_dv_nssec = base_pop_dv.aggregate(segs=["ns_sec"])

            # Read in base households
            base_hhs_dv = DVector.load(base_hhs_dv_path / f"Output P13.3_{gor}.hdf")
            base_hhs_dv_nssec = base_hhs_dv.aggregate(segs=["ns_sec"])

            # Read in forecast population
            forecast_dv = DVector.load(forecast_dv_path / f"Output Pop_{gor}_{forecast_year}.hdf")
            forecast_dv_nssec = forecast_dv.aggregate(segs=["ns_sec"])

            # Calculate base population NS-SEC % splits
            base_nssec = base_pop_dv_nssec.data
            base_total = base_nssec.sum()
            base_nssec_splits = (base_nssec / base_total)

            # Calculate forecast population NS-SEC % splits
            forecast_nssec = forecast_dv_nssec.data
            forecast_total = forecast_nssec.sum()
            forecast_nssec_splits = (forecast_nssec / forecast_total)

            # Calculate change in NS-SEC splits between base and forecast year for population
            split_change = forecast_nssec_splits - base_nssec_splits

            # Now apply these splits to the household data to get targets
            base_hh_nssec = base_hhs_dv_nssec.data
            base_hh_total = base_hh_nssec.sum()
            base_hh_nssec_splits = (base_hh_nssec / base_hh_total)

            new_hh_splits = base_hh_nssec_splits + split_change

            hh_nssec_targets = base_hh_total * new_hh_splits
            # Fill na (for the Scotland zones with zeroes)
            hh_nssec_targets = hh_nssec_targets.fillna(0)

            nssec_hh_targets.append(hh_nssec_targets)

        # Output and save as targets
        nssec_targets_output = pd.concat(nssec_hh_targets, axis=1)

        message = f"writing to {path_out}, with {hdf_key=}"
        print(message)
        nssec_targets_output.to_hdf(path_out / "hh_ns-sec_targets.hdf", key=hdf_key, mode="a")


def calc_and_output_hh_occupancy_targets(
        forecast_dv_path: Path,
        forecast_years: list,
        path_out: Path
):
    """
    Function to calculate occupancy targets for households, keeping sensible occupancies based on the in the
    outcome of the forecast population
    Keep separate for now as the occupancy household targets are based off the forecast population outputs

    Parameters
    ----------
    forecast_dv_path: Path
        Forecast population DVectors output
    forecast_years: list
        Years to generate forecasts for
    path_out: Path
        Location to output the DVector targets to
    """

    for forecast_year in forecast_years:
        occupancy_hh_targets_0 = []
        occupancy_hh_targets_1 = []
        occupancy_hh_targets_2 = []
        occupancy_hh_targets_3 = []
        hdf_key = f"targets_{forecast_year}"
        for gor in constants.GORS + ["Scotland"]:
            print(f"Processing for {forecast_year}, {gor}")
            # Read in forecast population
            forecast_dv = DVector.load(forecast_dv_path / f"Output Pop_{gor}_{forecast_year}.hdf")

            occ_targets = derive_household_occupancy_targets_forecasting(population_dvector=forecast_dv)

            # Targets are split out for four segmentations:
            # 1. adult population in 1 adult households with no children should match
            # number of households with 1 adult and no children
            # 2. adult population in 1 adult households with children should match
            # number of households with 1 adult and children
            # 3. adult population in 2 adult households with no children should be
            # double number of households with 2 adult and no children
            # 4. adult population in 2 adult households with children should be
            # double number of households with 2 adult and children
            targets_0 = occ_targets[0].data
            targets_1 = occ_targets[1].data
            targets_2 = occ_targets[2].data
            targets_3 = occ_targets[3].data

            occupancy_hh_targets_0.append(targets_0)
            occupancy_hh_targets_1.append(targets_1)
            occupancy_hh_targets_2.append(targets_2)
            occupancy_hh_targets_3.append(targets_3)

        # Output and save as targets
        household_adult_1_children_1_target = pd.concat(occupancy_hh_targets_0, axis=1)
        household_adult_1_children_2_target = pd.concat(occupancy_hh_targets_1, axis=1)
        household_adult_2_children_1_target = pd.concat(occupancy_hh_targets_2, axis=1)
        household_adult_2_children_2_target = pd.concat(occupancy_hh_targets_3, axis=1)

        message = f"writing to {path_out}, with {hdf_key=}"
        print(message)
        household_adult_1_children_1_target.to_hdf(path_out / "hh_occupancy_adult_1_children_1_target.hdf",
                                                   key=hdf_key, mode="a")
        household_adult_1_children_2_target.to_hdf(path_out / "hh_occupancy_adult_1_children_2_target.hdf",
                                                   key=hdf_key, mode="a")
        household_adult_2_children_1_target.to_hdf(path_out / "hh_occupancy_2_children_1_target.hdf",
                                                   key=hdf_key, mode="a")
        household_adult_2_children_2_target.to_hdf(path_out / "hh_occupancy_adult_2_children_2_target.hdf",
                                                   key=hdf_key, mode="a")


def filter_to_adults_forecasting(
        dvec: DVector,
        age_segmentation: str = 'age_ntem',
        adult_categories: Tuple[int] = (2, 3)
) -> DVector:
    """Filter a DVector to the adult age groups. This assumes that
    age_segmentation (e.g. age_9) is in the current segmentation of the DVector,
    otherwise this function will have no effect.

    Parameters
    ----------
    dvec: DVector
        Data to filter to only adult age groups. Assumed to have
        `adult_segmentation` in the dvec.segmentation.names, otherwise this
        function will return the original DVector.
    age_segmentation: str, default 'age_ntem'
        Segmentation name of the age category. Typically defined in SegmentsSuper.
    adult_categories: Tuple[int], default (2, 3)
        Values of the Segmentation that relate to the adult age groups of the
        age_segmentation name. Again, see SegmentsSuper for definitions.

    Returns
    -------
    DVector
        dvec with only `age_segmentation` categories `adult_categories`, or dvec
        if `age_segmentation` is not in dvec.segmentation.names.

    """

    # check if the segmentation is in the DVector
    if not age_segmentation in dvec.segmentation.names:
        LOGGER.warning(
            f'{age_segmentation} is not in the provided DVector. This function '
            f'will have no effect. Returning original DVector.'
        )
        return dvec

    return dvec.filter_segment_value(
        segment_name=age_segmentation, segment_values=list(adult_categories)
    )


def derive_household_occupancy_targets_forecasting(
        population_dvector: DVector,
        household_segments: tuple = (
                SegmentsSuper.ADULTS.value, SegmentsSuper.CHILDREN.value,
                SegmentsSuper.NS_SEC.value, SegmentsSuper.ACCOMODATION_TYPE_H.value
        ),
        children_segment_name: str = SegmentsSuper.CHILDREN.value,
        adult_segment_name: str = SegmentsSuper.ADULTS.value,
        no_children_hh_index: int = 1,
        yes_children_hh_index: int = 2,
        one_adult_hh_index: int = 1,
        two_adult_hh_index: int = 2
) -> list:
    """Derivation of household based targets for the IPF based on logical
    population and household linkages.

    This will derive IPF targets to ensure:
    - adult population in 1 adult households with no children should match
    number of households with 1 adult and no children
    - adult population in 1 adult households with children should match
    number of households with 1 adult and children
    - adult population in 2 adult households with no children should be
    double number of households with 2 adult and no children
    - adult population in 2 adult households with children should be
    double number of households with 2 adult and children


    Parameters
    ----------
    population_dvector: DVector
        Must have segmentation of *at least* household_segments. Represents
        total population.
    household_segments: tuple, default (
        SegmentsSuper.ADULTS.value, SegmentsSuper.CHILDREN.value,
        SegmentsSuper.NS_SEC.value, SegmentsSuper.ACCOMODATION_TYPE_H.value
        )
        Names of household segments to aggregate the population based targets to
    children_segment_name: str = SegmentsSuper.CHILDREN.value
        Name of the segmentation in population_dvector that represents the
        number of children in a household
    adult_segment_name: str = SegmentsSuper.ADULTS.value
        Name of the segmentation in population_dvector that represents the
        number of adults in a household
    no_children_hh_index: int = 1
        Segment value in children_segment_name that represents no children in
        the household
    yes_children_hh_index: int = 2
        Segment value in children_segment_name that represents yes children in
        the household
    one_adult_hh_index: int = 1
        Segment value in adult_segment_name that represents 1 adult in
        the household
    two_adult_hh_index: int = 2
        Segment value in adult_segment_name that represents 2 adults in
        the household

    Returns
    -------
    list
        Four DVectors that can be used as input targets to the IPF
    """
    targets = []

    # adult population in 1 adult households with no children should match
    # number of households with 1 adult and no children
    household_adult_1_children_1_target = filter_to_adults_forecasting(
        dvec=population_dvector
    ).filter_segment_value(
        segment_name=children_segment_name, segment_values=[no_children_hh_index]
    ).filter_segment_value(
        segment_name=adult_segment_name, segment_values=[one_adult_hh_index]
    ).aggregate(list(household_segments))
    household_adult_1_children_1_target.data = household_adult_1_children_1_target.data.replace(
        to_replace=0, value=0.000000000001
    )

    # adult population in 1 adult households with children should match
    # number of households with 1 adult and children
    household_adult_1_children_2_target = filter_to_adults_forecasting(
        dvec=population_dvector
    ).filter_segment_value(
        segment_name=children_segment_name, segment_values=[yes_children_hh_index]
    ).filter_segment_value(
        segment_name=adult_segment_name, segment_values=[one_adult_hh_index]
    ).aggregate(list(household_segments))
    household_adult_1_children_2_target.data = household_adult_1_children_2_target.data.replace(
        to_replace=0, value=0.000000000001
    )

    # adult population in 2 adult households with no children should be
    # double number of households with 2 adult and no children
    household_adult_2_children_1_target = (filter_to_adults_forecasting(
        dvec=population_dvector
    ).filter_segment_value(
        segment_name=children_segment_name, segment_values=[no_children_hh_index]
    ).filter_segment_value(
        segment_name=adult_segment_name, segment_values=[two_adult_hh_index]
    ) / 2).aggregate(list(household_segments))
    household_adult_2_children_1_target.data = household_adult_2_children_1_target.data.replace(
        to_replace=0, value=0.000000000001
    )

    # adult population in 2 adult households with children should be
    # double number of households with 2 adult and children
    household_adult_2_children_2_target = (filter_to_adults_forecasting(
        dvec=population_dvector
    ).filter_segment_value(
        segment_name=children_segment_name, segment_values=[yes_children_hh_index]
    ).filter_segment_value(
        segment_name=adult_segment_name, segment_values=[two_adult_hh_index]
    ) / 2).aggregate(list(household_segments))
    household_adult_2_children_2_target.data = household_adult_2_children_2_target.data.replace(
        to_replace=0, value=0.000000000001
    )

    return [
        household_adult_1_children_1_target, household_adult_1_children_2_target,
        household_adult_2_children_1_target, household_adult_2_children_2_target
        ]


# %%
def fetch_ons_pop_forecasts_up_to_crossover(year: int) -> pd.DataFrame:

    gor_2018, country_2018 = fetch_2018_pop_projections(year=year)
    country_2022 = fetch_2022_pop_projections(year=year)

    # adjust based on more recent country level data
    update_forecast_factor = country_2022 / country_2018
    return gor_2018 * update_forecast_factor


def fetch_ons_pop_forecasts_post_crossover(year: int) -> pd.DataFrame:
    # As regional 2018 forecasts aren't available after 2043 then a different approach is needed
    gor_2018_crossover, country_2018_crossover = fetch_2018_pop_projections(
        year=CROSSOVER_YEAR
    )
    country_2022_crossover = fetch_2022_pop_projections(year=CROSSOVER_YEAR)
    country_2022_fy = fetch_2022_pop_projections(year=year)

    # adjust from crossover to forecast year
    update_forecast_factor = country_2022_crossover / country_2018_crossover
    crossover_to_fy_factor = country_2022_fy / country_2022_crossover
    return gor_2018_crossover * update_forecast_factor * crossover_to_fy_factor


# %%
def fetch_2018_pop_projections(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Note only England has projections by sub areas
    # however both Scotland and Wales are themselves regions
    eng_regional_forecast_filepath = (
        POPULATION_DIR / "2018_based_england_regions_pop_projections.xlsx"
    )

    gor_df = pd.read_excel(
        eng_regional_forecast_filepath, sheet_name="Males", skiprows=6
    )
    male_df = convert_rgn_forecasts_to_segmentations(df=gor_df, g_seg=1)
    male_df_rgn = male_df[male_df["RGN2021"] != "E92000001"]
    male_df_country = create_english_totals_from_regional(df=male_df)

    gor_df = pd.read_excel(
        eng_regional_forecast_filepath, sheet_name="Females", skiprows=6
    )
    female_df = convert_rgn_forecasts_to_segmentations(df=gor_df, g_seg=2)
    female_df_rgn = female_df[female_df["RGN2021"] != "E92000001"]
    female_df_country = create_english_totals_from_regional(df=female_df)

    scotland_forecast = POPULATION_DIR / "2022_based_scotland_pop_projections.xlsx"
    df_age_g_scotland = process_to_segmentations(path_in=scotland_forecast)
    df_age_g_scotland["RGN2021"] = "S92000003"

    wales_forecast = POPULATION_DIR / "2021_based_interim_wales_pop_projections.xlsx"
    df_age_g_wales = process_to_segmentations(path_in=wales_forecast)
    df_age_g_wales["RGN2021"] = "W92000004"

    gor_df = pd.concat([male_df_rgn, female_df_rgn, df_age_g_scotland, df_age_g_wales])
    country_df = pd.concat(
        [male_df_country, female_df_country, df_age_g_scotland, df_age_g_wales]
    )

    if year not in gor_df.columns:
        raise NotImplementedError(f"{year} is not included in dataset")

    df_gor_wide = pd.pivot(
        gor_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
    )

    df_country_wide = pd.pivot(
        country_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
    )

    return df_gor_wide, df_country_wide


def create_english_totals_from_regional(df: pd.DataFrame) -> pd.DataFrame:

    # find regions mentioned
    eng_regions_list = df["RGN2021"].unique().tolist()

    # remove england from the list
    eng_regions_list.remove(ENGLAND_CODE)
    eng_values = df[df["RGN2021"] == ENGLAND_CODE]

    regional_dfs = []
    for eng_region in eng_regions_list:
        df_current = eng_values.copy()
        df_current["RGN2021"] = eng_region
        regional_dfs.append(df_current)

    return pd.concat(regional_dfs)


def convert_rgn_forecasts_to_segmentations(
    df: pd.DataFrame, g_seg: int
) -> pd.DataFrame:

    df = df.rename(columns={"CODE": "RGN2021"})

    df = df.drop(columns="AREA")

    # get lowest age
    df = allocate_age_ntem(df)

    df = df.groupby(["RGN2021", "age_ntem"]).sum()
    df["g"] = g_seg

    df = df.reset_index()
    return df


def allocate_age_ntem(df: pd.DataFrame) -> pd.DataFrame:
    df["from_age"] = df["AGE GROUP"].str.split("-").str[0]

    # fix 15-19 issue mapping to two categories in ratio 1:4
    df_exc_15 = df[df["from_age"] != "15"].copy()
    # don't forget about 90+
    df_exc_15["from_age"] = df_exc_15["from_age"].str.replace("+", "")

    # drop All Ages
    df_exc_15 = df_exc_15[df_exc_15["from_age"] != "All ages"]
    # convert to int
    df_exc_15["from_age"] = df_exc_15["from_age"].astype(int)

    df_15 = df[df["from_age"] == "15"].copy()
    df_15["from_age"] = 15

    available_years = list(range(2018, 2044))
    df_15[available_years] = df_15[available_years] * 0.2

    df_16 = df[df["from_age"] == "15"].copy()
    df_16["from_age"] = 16
    df_16[available_years] = df_16[available_years] * 0.8

    # and combine
    df = pd.concat([df_exc_15, df_15, df_16])

    df = find_age_ntem_enum(df=df, age_col="from_age")

    df = df.drop(columns=["AGE GROUP", "from_age"])

    return df


# %%
def find_age_ntem_enum(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df.loc[df[age_col] < 16, "age_ntem"] = 1  # aged 15 years and under
    df.loc[(df[age_col] >= 16) & (df[age_col] <= 74), "age_ntem"] = (
        2  # aged 16 to 74 years
    )
    df.loc[df[age_col] >= 75, "age_ntem"] = 3  # aged 75 and over
    df["age_ntem"] = df["age_ntem"].astype(int)
    return df


def fetch_region_correspondence() -> pd.DataFrame:

    return pd.read_csv(
        Path(
            r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
            "GOR2021_CD_NM_EWS.csv",
        )
    )


# %%
def fetch_2022_pop_projections(year: int) -> pd.DataFrame:

    country_forecasts = {}

    england_forecast = (
        POPULATION_DIR / "2021_based_interim_england_pop_projections.xlsx"
    )
    df_age_g_eng = process_to_segmentations(path_in=england_forecast)
    country_forecasts["england"] = df_age_g_eng

    # note this is 2022 whereas england and wales are 2021
    scotland_forecast = POPULATION_DIR / "2022_based_scotland_pop_projections.xlsx"
    df_age_g_scotland = process_to_segmentations(path_in=scotland_forecast)
    country_forecasts["scotland"] = df_age_g_scotland

    wales_forecast = POPULATION_DIR / "2021_based_interim_wales_pop_projections.xlsx"
    df_age_g_wales = process_to_segmentations(path_in=wales_forecast)
    country_forecasts["wales"] = df_age_g_wales

    region_correspondence = pd.read_csv(
        Path(
            r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
            "GOR2021_CD_NM_EWS.csv",
        )
    )
    region_codes = region_correspondence["RGN21CD"]

    regions_df = create_2022_regions_df(
        country_forecasts=country_forecasts,
        region_codes=region_codes,
    )

    country_pop_projections = pd.pivot(
        regions_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
    )

    return country_pop_projections


# %%
def process_to_segmentations(path_in: Path) -> pd.DataFrame:
    df = pd.read_excel(path_in, sheet_name="Population")

    # for our purposes we can map '105 - 109' and '110 and over' to 105

    df.loc[df["Age"] == "105 - 109", "Age"] = 105
    df.loc[df["Age"] == "110 and over", "Age"] = 105

    df["Age"] = df["Age"].astype(int)

    df = find_age_ntem_enum(df=df, age_col="Age")

    # and now allocate g segment
    df.loc[df["Sex"] == "Males", "g"] = 1
    df.loc[df["Sex"] == "Females", "g"] = 2
    df["g"] = df["g"].astype(int)

    df = df.drop(columns=["Sex", "Age"])

    df = df.groupby(["age_ntem", "g"]).sum()
    return df.reset_index()


# %% region approach
def create_2022_regions_df(
    country_forecasts: dict[str, pd.DataFrame],
    region_codes: list[str],
) -> pd.DataFrame:

    # if region approach with multiple columns for the data then go down this path

    # note that we duplicate the values for england to each region, not ideal but
    # as only going to look for growth from 2023 should be fine as long
    # as we don't use this value directly, and do it consistently for the base and future year

    region_list = []
    for region_code in region_codes:
        if region_code.startswith("E"):
            current_df = country_forecasts["england"]
        elif region_code.startswith("S"):
            current_df = country_forecasts["scotland"]
        elif region_code.startswith("W"):
            current_df = country_forecasts["wales"]
        else:
            raise ValueError(f"Unable to process {region_code}")

        region_df = current_df.copy()
        region_df["RGN2021"] = region_code
        region_list.append(region_df)

    regions_df = pd.concat(region_list)

    return regions_df.reset_index()


def process_and_save_hh_projections() -> None:
    """
    Function to read in and pre-process the 2018-based ONS households
    projections
    Outputs the totals household projections by GOR for each year, as hdfs

    Can substitute other sources of data for this input (e.g. NTEM) if required
    """
    rgn_corr = fetch_region_correspondence()

    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / "england_LAs_2018_based_hh_projections.xlsx",
        sheet_name="406",
        header=4,
    ).rename(columns={"Area code": "region"})
    hh_eng = hh_eng.iloc[:, np.r_[0:2, 19:45]]
    # Filter to the defined regions (2018-based ONS data is available up to 2043)
    hh_eng = hh_eng[hh_eng["Area name"].isin(rgn_corr["RGN21NM"].tolist())].drop(
        columns=["Area name"]
    )

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / "scotland_2018-based_hh_projections.xlsx",
        sheet_name="Table 1",
        header=3,
        nrows=1,
    )
    hh_scot = hh_scot.iloc[:, 1:27]
    hh_scot.columns = hh_scot.columns.astype(int)
    # Define region code
    hh_scot["region"] = "S92000003"

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / "wales_2018_based_hh_projections.csv", header=5, nrows=1
    )
    hh_wales = hh_wales.drop(hh_wales.columns[0:2], axis=1)
    hh_wales.columns = hh_wales.columns.str.strip().astype(int)
    # Define region code
    hh_wales["region"] = "W92000004"

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    hh_projs = pp.infill_for_years(
        df=hh_projs, forecast_years=FORECAST_YEARS, extroplate_beyond_end="trend"
    )

    # Export
    for year in FORECAST_YEARS:

        hh_projs[f"factors_from_{BASE_YEAR}"] = hh_projs[year] / hh_projs[BASE_YEAR]

        # Create column to use as segmentation totals
        hh_projs["total"] = 1

        df_wide = pd.pivot(
            hh_projs,
            index=["total"],
            columns=["region"],
            values=f"factors_from_{BASE_YEAR}",
        )

        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / f"hh_totals_from_{BASE_YEAR}_factors.hdf",
            df=df_wide,
            key=f"factors_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )


def process_and_save_hh_projections_children() -> None:
    """
    Function to read in and pre-process the 2018-based ONS households
    projections
    Outputs the children household projections by England GOR for each year, as hdfs
    1: Household with no children or all children non-dependent
    2: Household with one or more dependent children

    Can substitute other sources of data for this input (e.g. NTEM) if required

    """
    rgn_corr = fetch_region_correspondence()

    # Household projections 2023 to 2043 (all years)
    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / "eng_2018_based_Stage 2 projected households - Principal.xlsx",
        sheet_name="Households by type",
    )

    # Filter to the regions and "all ages"
    hh_eng = hh_eng[hh_eng["AREA NAME"].isin(rgn_corr["RGN21NM"].tolist())]
    hh_eng = hh_eng.loc[hh_eng["AGE GROUP"] == "All ages"]
    hh_eng = hh_eng.loc[hh_eng["HOUSEHOLD TYPE"] != "Total"]
    years = [year for year in range(2018, 2044)]
    hh_eng = hh_eng.loc[:, ["CODE", "HOUSEHOLD TYPE"] + years]

    # Map to Land Use children segmentation (hh with no children / hh with 1+ children)
    hh_eng["segment"] = hh_eng["HOUSEHOLD TYPE"].map(CHILDREN_MAPPING).astype(int)
    hh_eng = hh_eng.groupby(by=["CODE", "segment"], as_index=False)[years].sum()

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / "scotland_2018-based_hh_projections.xlsx",
        sheet_name="Table 2",
        header=3,
        nrows=7,
    )
    hh_scot = hh_scot.iloc[:, 1:28]
    hh_scot.columns = [int(col) if col.isdigit() else col for col in hh_scot.columns]
    hh_scot = hh_scot.loc[:, ["Household type"] + years]

    # Define region code
    hh_scot["CODE"] = "S92000003"

    # Map to Land Use children segmentation (hh with no children / hh with 1+ children)
    hh_scot["segment"] = hh_scot["Household type"].map(CHILDREN_MAPPING).astype(int)
    hh_scot = hh_scot.groupby(by=["CODE", "segment"], as_index=False)[years].sum()

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / "wales_2018_based_hh_projections.csv", header=5, nrows=13
    )
    hh_wales = (
        hh_wales.drop(hh_wales.columns[0], axis=1)
        .drop(hh_wales.index[0], axis=0)
        .rename(columns={hh_wales.columns[1]: "Household type"})
    )
    hh_wales.columns = hh_wales.columns.str.strip()
    hh_wales.columns = [int(col) if col.isdigit() else col for col in hh_wales.columns]
    hh_wales["Household type"] = hh_wales["Household type"].str.rstrip()
    # Define region code
    hh_wales["CODE"] = "W92000004"

    # Map to Land Use children segmentation
    # (hh with no children / hh with 1+ children)
    hh_wales["segment"] = hh_wales["Household type"].map(CHILDREN_MAPPING).astype(int)
    hh_wales = hh_wales.groupby(by=["CODE", "segment"], as_index=False)[years].sum()

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    hh_projs = pp.infill_for_years(
        df=hh_projs, forecast_years=FORECAST_YEARS, extroplate_beyond_end="trend"
    )

    hh_projs = hh_projs.rename(columns={"segment": "children"})
    hh_projs.columns = [str(col) for col in hh_projs.columns]

    for year in FORECAST_YEARS:

        hh_projs[f"factors_{year}"] = hh_projs[str(year)] / hh_projs[str(BASE_YEAR)]

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=hh_projs,
            zoning_column="CODE",
            index_cols=["children"],
            value_column=f"factors_{year}",
        )
        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR
            / f"hh_children_from_{BASE_YEAR}_factors.hdf",
            df=df_wide,
            key=f"factors_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )


def pre_process_obr() -> None:
    # TODO: do we actually use this?

    # Read in OBR data (using quarter 1 data)
    df = pd.read_csv(OBR_INPUT_DIR / "obr_forecasts_feb_2025.csv")

    # employment
    emp = df.loc[
        (df["Measure"] == "Employment forecast")
        | (df["Measure"] == "Employment outturn")
    ]
    emp = emp[emp["Quarter"].str.endswith("Q1")].sort_values("Quarter")
    emp["year"] = emp["Quarter"].str.replace("Q1", "").astype(int)
    emp["Measure"] = "employment"

    # unemployment (percentage pf population 16+)
    unemp = df.loc[
        (df["Measure"] == "Unemployment rate forecast")
        | (df["Measure"] == "Unemployment rate outturn")
    ]
    unemp = unemp[unemp["Quarter"].str.endswith("Q1")].sort_values("Quarter")
    unemp["year"] = unemp["Quarter"].str.replace("Q1", "").astype(int)
    unemp["Measure"] = "unemployment"

    both = pd.concat([emp, unemp], ignore_index=False)[["year", "Measure", "Value"]]
    both = both.pivot(index="Measure", columns="year", values="Value")

    both = pp.infill_for_years(
        df=both, forecast_years=FORECAST_YEARS, extroplate_beyond_end="trend"
    )

    # Prepare for export
    export = both[FORECAST_YEARS]
    output_folder = OBR_INPUT_DIR / "preprocessing"

    output_folder.mkdir(exist_ok=True)
    export.to_csv(
        output_folder / "OBR_forecasts_processed.csv",
        index=True,
        index_label="Measure",
    )


def process_and_save_projections_1_adult_hhs():
    # This function processes the households data into 1 adult by gender and not by gender, by local authority (2019)

    rgn_corr = fetch_region_correspondence()
    years = [year for year in range(2018, 2044)]

    # -------------
    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR
        / "eng_2018_based_Stage 2 projected households - Principal (2019 geog).xlsx",
        sheet_name="Households by type",
    )
    # Filter to the local authorities, "all ages" and 1-person households
    hh_eng = hh_eng[
        ~hh_eng["AREA NAME"].isin(rgn_corr["RGN21NM"].tolist())
        & ~hh_eng["AREA NAME"].isin(["England"])
    ]
    hh_eng = hh_eng.loc[hh_eng["AGE GROUP"] == "All ages"]
    hh_eng = hh_eng.loc[
        (hh_eng["HOUSEHOLD TYPE"] == "One person households: Male")
        | (hh_eng["HOUSEHOLD TYPE"] == "One person households: Female")
    ]
    hh_eng = hh_eng.loc[:, ["CODE", "HOUSEHOLD TYPE"] + years]

    # Map to Land Use gender segmentation
    hh_eng["g"] = (
        hh_eng["HOUSEHOLD TYPE"]
        .map({"One person households: Male": 1, "One person households: Female": 2})
        .astype(int)
    )
    hh_eng = hh_eng.groupby(by=["CODE", "g"], as_index=False)[years].sum()

    # -------------
    # SCOTLAND
    scot_la_codes = pd.read_csv(
        HOUSEHOLDS_DIR / "scot_LA_codes.csv", usecols=["lad19cd", "lad19nm"]
    )
    scot_dfs = []
    for code in scot_la_codes["lad19nm"]:
        hh_scot = pd.read_excel(
            HOUSEHOLDS_DIR
            / "scotland_2018_based_hh_detailed-area-tables-principal-projection.xlsx",
            sheet_name=code,
            header=4,
            nrows=34,
        )
        hh_scot = hh_scot.rename(
            columns={hh_scot.columns[0]: "Household type", hh_scot.columns[1]: "age"}
        )
        hh_scot = hh_scot.drop(hh_scot.columns[28:34], axis=1)
        hh_scot = hh_scot[hh_scot["age"] != "All Ages"]
        hh_scot["region"] = code
        hh_scot["CODE"] = hh_scot["region"].map(
            scot_la_codes.set_index("lad19nm")["lad19cd"].to_dict()
        )
        hh_scot["g"] = (
            hh_scot["Household type"]
            .map({"1 person male": 1, "1 person female": 2})
            .astype(int)
        )
        hh_scot.columns = [
            int(col) if col.isdigit() else col for col in hh_scot.columns
        ]
        hh_scot = hh_scot.groupby(by=["CODE", "g"], as_index=False)[years].sum()
        scot_dfs.append(hh_scot)
    hh_scot = pd.concat(scot_dfs)

    # -------------
    # WALES
    wales_dfs = []
    for g in ["males", "females"]:
        hh_wales = pd.read_csv(
            HOUSEHOLDS_DIR / f"wales_2018_based_hh_projections_LAs_1_person_hh_{g}.csv",
            header=8,
            nrows=22,
        )
        hh_wales = hh_wales.rename(columns={hh_wales.columns[0]: "Household type"})
        hh_wales["Household type"] = hh_wales["Household type"].str.rstrip()

        wales_la_codes = pd.read_csv(
            HOUSEHOLDS_DIR / "wales_LA_codes.csv", usecols=["lad19cd", "lad19nm"]
        )
        hh_wales = pd.merge(
            hh_wales,
            wales_la_codes,
            left_on="Household type",
            right_on="lad19nm",
            how="left",
        ).rename(columns={"lad19cd": "CODE"})
        if g == "males":
            hh_wales["g"] = 1
        elif g == "females":
            hh_wales["g"] = 2
        wales_dfs.append(hh_wales)
    hh_wales = pd.concat(wales_dfs)
    hh_wales.columns = hh_wales.columns.str.strip()
    hh_wales.columns = [int(col) if col.isdigit() else col for col in hh_wales.columns]
    hh_wales = hh_wales.groupby(by=["CODE", "g"], as_index=False)[years].sum()

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    hh_projs = pp.infill_for_years(
        df=hh_projs, forecast_years=FORECAST_YEARS, extroplate_beyond_end="trend"
    )

    hh_projs["adults"] = 1
    hh_projs.columns = [str(col) for col in hh_projs.columns]

    columns_to_sum = [col for col in hh_projs.columns if col.isdigit()]
    hh_projs_by_g = hh_projs.copy()
    hh_projs_no_g = hh_projs.groupby(by=["CODE", "adults"], as_index=False)[columns_to_sum].sum()

    for year in FORECAST_YEARS:
        for x in ["g", "no_g"]:
            if x == "g":
                hh_projs_output = hh_projs_by_g.copy()
                index_columns = ["g", "adults"]
            else:
                hh_projs_output = hh_projs_no_g.copy()
                index_columns = ["adults"]

            hh_projs_output[f"{year}_factor"] = hh_projs_output[str(year)] / hh_projs_output[str(BASE_YEAR)]

            # Into a wide format for DVector
            hh_projs_wide = pp.pivot_to_dvector(
                data=hh_projs_output,
                zoning_column="CODE",
                index_cols=index_columns,
                value_column=f"{year}_factor",
            )
            pp.save_preprocessed_hdf(
                source_file_path=HOUSEHOLDS_DIR
                / f"hh_1_adult_by_{x}_from_{BASE_YEAR}_factors.hdf",
                df=hh_projs_wide,
                key=f"factors_from_{BASE_YEAR}_to_{year}",
                mode="a",
            )


def process_base_to_ntem_age(rgn: str):
    input_folder = Path(r"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs")
    output_folder = Path(r"F:\Working\Land-Use\BASE_POPULATION_WITH_AGE_NTEM\based_on_241220_Populationv2")
    output_folder.mkdir(exist_ok=True)

    base_pop_path = input_folder / f"Output P11_{rgn}.hdf"
    if os.path.exists(output_folder / base_pop_path.name):
        print(f"Base population file for {rgn} has already been processed to age_ntem and exists in folder")
    else:
        print(f"Base population file for {rgn} does not exist, generating now...")

        base_pop = DVector.load(base_pop_path)

        base_pop_ntem_age = base_pop.translate_segment(
            from_seg="age_9", to_seg="age_ntem", drop_from=True
        )
        out_path = output_folder / base_pop_path.name

        base_pop_ntem_age.save(out_path)


# %%
if __name__ == "__main__":
    main()
