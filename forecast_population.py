# %%
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import shutil

import yaml

from caf.base.data_structures import DVector
from caf.base.segmentation import SegmentsSuper

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel


def fetch_base_pop(config: dict, gor: str) -> DVector:
    """
    Function to fetch the base population, using population directory defined in the config file

    Parameters
    ----------
    config: dictionary
        config file including the base population path and file name
    gor: str
        region to fetch base population for, as base outputs are separated by region
    """
    # --- Step 0 --- #
    # Load in the Base population output
    LOGGER.info("--- Step 0 ---")
    LOGGER.info("Load in the Base population output")

    base_pop_directory = Path(config["base_data"]["pop_output_directory"])
    base_pop_file_stem = config["base_data"]["base_pop_file_stem"]
    filepath = base_pop_directory / f"{base_pop_file_stem}_{gor}.hdf"
    base_pop = DVector.load(filepath)

    base_seg_totals = data_processing.find_segment_totals(
        dvec=base_pop, dimension="population"
    )

    dir_out = OUTPUT_DIR / OutputLevel.ASSURANCE
    dir_out.mkdir(parents=True, exist_ok=True)
    path_out = dir_out / "pop_base_segment_totals.csv"

    base_seg_totals.to_csv(
        path_out,
        float_format="%.5f",
        index=False,
    )

    # # and for the regional ones
    # find_regional_seg_totals(dvec=base_pop, output_prefix="pop_base_segment_totals")

    return base_pop


def fetch_base_households(config: dict, gor: str) -> DVector:
    """
    Function to fetch the base households, using households directory defined in the config file

    Parameters
    ----------
    config: dictionary
        config file including the base households path and file name
    gor: str
        region to fetch base households for, as base outputs are separated by region
    """
    # --- Step 0 --- #
    # Load in the Base households output
    LOGGER.info("--- Step 0 ---")
    LOGGER.info("Load in the Base households output")
    base_households_directory = Path(config["base_data"]["household_output_directory"])
    base_hh_file_stem = config["base_data"]["base_households_file_stem"]
    filepath = base_households_directory / f"{base_hh_file_stem}_{gor}.hdf"
    base_hhs = DVector.load(filepath)

    base_seg_totals = data_processing.find_segment_totals(
        dvec=base_hhs, dimension="households"
    )

    dir_out = OUTPUT_DIR / OutputLevel.ASSURANCE
    dir_out.mkdir(parents=True, exist_ok=True)
    path_out = dir_out / "households_base_segment_totals.csv"

    base_seg_totals.to_csv(
        path_out,
        float_format="%.5f",
        index=False,
    )

    # # and for the regional ones
    # find_regional_seg_totals(dvec=base_hhs, output_prefix="households_base_segment_totals")

    return base_hhs


def fetch_base_occs(config: dict, gor: str) -> DVector:
    """
    Function to fetch the base occupancies, using population directory defined in the config file

    Parameters
    ----------
    config: dictionary
        config file including the base population path and file name
    gor: str
        region to fetch base occupancies for, as base outputs are separated by region
    """
    # --- Step 0 --- #
    # Load in the Base population output
    LOGGER.info("--- Step 0 ---")
    LOGGER.info("Load in the Base occupancies output")

    base_occs_directory = Path(config["base_data"]["occs_output_directory"])
    base_occs_file_stem = config["base_data"]["base_occs_file_stem"]
    filepath = base_occs_directory / f"{base_occs_file_stem}_{gor}.hdf"
    base_occs = DVector.load(filepath)

    return base_occs


# %%
def process_forecast_pop_by_gor(
        config: dict,
        base_pop: DVector,
        forecast_year: int,
        gor: str,
        target_dvector_key: int,
        maintain_base_distributions: bool,
        segments_to_maintain: Optional[list] = None) -> None:
    """
    Function to process and generate forecast population by regions
    Reads in from the config file, with IPF targets defined in the config

    Parameters
    ----------
    config: dictionary
        config file including the targets (e.g. population projections by age, soc projections)
    base_pop: DVector
        base population DVector that has been read in previously using "fetch_base_pop" function
    forecast_year: int
        year to calculate forecasts for
    gor: str
        region to calculate forecasts for
    target_dvector_key: int
        defines which Dvector that is read in as the targets to take as the total target DVector
    maintain_base_distributions: bool
        whether to include maintaining base distributions in the IPF targets (e.g. maintaining NS-SEC distributions)
    segments_to_maintain: Optional[list]
        option defining which segments to maintain from the base, defined in the config
    """

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Load in IPF targets")

    pop_targets = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="population_targets",
        geography_subset=gor,
        hdf_key=f"targets_{forecast_year}",
    )

    # fix target_dvectors to be a list
    if isinstance(pop_targets, list):
        target_dvectors = pop_targets
    else:
        target_dvectors = [pop_targets]

    # Get base distribution targets if maintaining
    # TODO refactor and add warning
    if maintain_base_distributions:
        added_targets = []
        for segmentation in segments_to_maintain:
            if isinstance(segmentation, list):
                target = base_pop.aggregate(segs=segmentation)
            else:
                target = base_pop.aggregate(segs=[segmentation])

            added_targets.append(target)

        # TODO switch the order here?
        target_dvectors = target_dvectors + added_targets

        match_totals_dvector = target_dvectors[target_dvector_key]

    else:
        if target_dvector_key == "None":
            match_totals_dvector = None
        else:
            match_totals_dvector = target_dvectors[target_dvector_key]

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")

    # Apply the IPF to targets
    LOGGER.info("Apply the IPF to targets")
    rebalanced_pop, summary, differences = data_processing.apply_ipf(
        seed_data=base_pop,
        target_dvectors=target_dvectors,
        cache_folder=constants.CACHE_FOLDER,
        # use pop targets as the target dvector to "match totals to"
        target_dvector=match_totals_dvector
    )

    # Add "total" segmentation for consistency of outputs
    if "total" not in rebalanced_pop.segmentation.names:
        rebalanced_pop = rebalanced_pop.add_segments([SegmentsSuper.get_segment(SegmentsSuper.TOTAL)])

    output_reference = f"Output Pop_{gor}_{forecast_year}"
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=output_reference,
        dvector=rebalanced_pop,
        dvector_dimension="people",
        output_level=OutputLevel.FINAL,
    )

    (OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}").mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}" / f"{output_reference}_VALIDATION.csv",
        float_format="%.5f",
        index=False,
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}",
        file=f"{output_reference}_VALIDATION.xlsx",
        dfs=differences,
    )

    forecast_seg_totals = data_processing.find_segment_totals(
        dvec=rebalanced_pop, dimension="population"
    )
    forecast_seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE / f"{gor}"
        / f"{output_reference}_segment_totals.csv",
        float_format="%.5f",
        index=False
    )


def process_forecast_households_by_gor(
        config: dict,
        base_households: DVector,
        forecast_year: int,
        gor: str,
        target_dvector_key: int,
        maintain_base_distributions: bool,
        segments_to_maintain: Optional[list] = None
):
    """
    Function to process and generate forecast households by regions
    Reads in from the config file, with IPF targets defined in the config

    Parameters
    ----------
    config: dictionary
        config file including the targets (e.g. total household projections)
    base_households: DVector
        base households DVector that has been read in previously using "fetch_base_households" function
    forecast_year: int
        year to calculate forecasts for
    gor: str
        region to calculate forecasts for
    target_dvector_key: int
        defines which Dvector that is read in as the targets to take as the total target DVector
    maintain_base_distributions: bool
        whether to include maintaining base distributions in the IPF targets (e.g. maintaining NS-SEC distributions)
    segments_to_maintain: Optional[list]
        option defining which segments to maintain from the base, defined in the config
    """
    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Load in IPF targets")

    household_targets = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="household_targets",
        geography_subset=gor,
        hdf_key=f"targets_{forecast_year}",
    )

    # fix target_dvectors to be a list
    if isinstance(household_targets, list):
        target_dvectors = household_targets
    else:
        target_dvectors = [household_targets]

    # Get base distribution targets if maintaining
    if maintain_base_distributions:
        added_targets = []
        for segmentation in segments_to_maintain:
            if isinstance(segmentation, list):
                target = base_pop.aggregate(segs=segmentation)
            else:
                target = base_pop.aggregate(segs=[segmentation])

            added_targets.append(target)

        # TODO switch the order here?
        target_dvectors = target_dvectors + added_targets

        match_totals_dvector = target_dvectors[target_dvector_key]

    else:
        if target_dvector_key == "None":
            match_totals_dvector = None
        else:
            match_totals_dvector = target_dvectors[target_dvector_key]

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")

    # Apply the IPF to targets
    LOGGER.info("Apply the IPF to targets")
    rebalanced_hhs, summary, differences = data_processing.apply_ipf(
        seed_data=base_households,
        target_dvectors=target_dvectors,
        cache_folder=constants.CACHE_FOLDER,
        # use household totals targets as the target dvector to "match totals to"
        target_dvector=match_totals_dvector
    )

    output_reference = f"Output Households_{gor}_{forecast_year}"
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=output_reference,
        dvector=rebalanced_hhs,
        dvector_dimension="households",
        output_level=OutputLevel.FINAL,
    )

    (OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}").mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}" / f"{output_reference}_VALIDATION.csv",
        float_format="%.5f",
        index=False,
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}" ,
        file=f"{output_reference}_VALIDATION.xlsx",
        dfs=differences,
    )

    forecast_seg_totals = data_processing.find_segment_totals(
        dvec=rebalanced_hhs, dimension="households"
    )
    forecast_seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE / f"{gor}"
        / f"{output_reference}_segment_totals.csv",
        float_format="%.5f",
        index=False,
    )


def process_forecast_households_based_on_pop(
        base_occs: DVector,
        forecast_year: int,
        gor: str
):
    """
    Function to process and generate forecast households by regions, based on the forecast population
    Reads in from the config file, with IPF targets defined in the config

    Parameters
    ----------
    base_occs: DVector
        base occupancies DVector that has been read in previously using "fetch_base_occs" function
    forecast_year: int
        year to calculate forecasts for
    gor: str
        region to calculate forecasts for
    """

    pop_output_reference = f"Output Pop_{gor}_{forecast_year}.hdf"
    forecast_pop = DVector.load(OUTPUT_DIR / OutputLevel.FINAL / pop_output_reference)
    # Aggregate forecast population to same segmentation as base occupancies
    forecast_pop_agg = forecast_pop.aggregate(
        [nom for nom in base_occs.segmentation.names if nom in forecast_pop.segmentation.names]
    )

    forecast_hhs = forecast_pop_agg / base_occs

    # Check for undefined values
    if forecast_hhs.data.isnull().values.any() is True:
        LOGGER.warning('Undefined values detected in the forecast household data')

    output_reference = f"Output Households_{gor}_{forecast_year}"
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=output_reference,
        dvector=forecast_hhs,
        dvector_dimension="households",
        output_level=OutputLevel.FINAL,
    )

    (OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{gor}").mkdir(parents=True, exist_ok=True)

    forecast_seg_totals = data_processing.find_segment_totals(
        dvec=forecast_hhs, dimension="households"
    )
    forecast_seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE / f"{gor}"
        / f"{output_reference}_segment_totals.csv",
        float_format="%.5f",
        index=False,
    )


# %%
# python forecast_population.py "path/to_file/forecast_population_config.yml"

# TODO: expand on the documentation here
parser = ArgumentParser("Land-Use forecast population command line runner")
parser.add_argument("config_file", type=Path)
args = parser.parse_args()

with open(args.config_file, "r") as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

# Get output directory for intermediate outputs from config file
OUTPUT_DIR = Path(config["output_directory"])
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = bool(config["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="population"
)

# copy config file for traceability
shutil.copy(
    src=args.config_file,
    dst=OUTPUT_DIR / OutputLevel.SUPPORTING / args.config_file.name,
)


base_year = config["base_year"]
forecast_years = config["forecast_years"]
run_for_regions = config["run_for_regions"]

for region in run_for_regions:
    base_pop = fetch_base_pop(config=config, gor=region)
    base_hhs = fetch_base_households(config=config, gor=region)
    base_occs = fetch_base_occs(config=config, gor=region)

    for forecast_year in forecast_years:
        process_forecast_pop_by_gor(
            config=config, base_pop=base_pop, forecast_year=forecast_year, gor=region,
            maintain_base_distributions=config["maintain_population_base_distributions"],
            target_dvector_key=config["forecast_population_total_target_key"]
        )

        if config["forecast_data"]["household_targets"] is None:
            # Calculate households based off the forecast population and base occupancies
            process_forecast_households_based_on_pop(
                base_occs=base_occs, forecast_year=forecast_year, gor=region
            )

        else:
            # Calculate households using the IPF
            process_forecast_households_by_gor(
                config=config, base_households=base_hhs, forecast_year=forecast_year, gor=region,
                maintain_base_distributions=config["maintain_households_base_distributions"],
                target_dvector_key=config["forecast_household_total_target_key"]
            )
