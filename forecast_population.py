# %%
from argparse import ArgumentParser
from pathlib import Path
import shutil

import yaml

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel


def fetch_base_pop(config: dict, gor: str) -> DVector:
    # --- Step 0 --- #
    # Load in the Base population output
    LOGGER.info("--- Step 0 ---")
    LOGGER.info("Load in the Base population output")

    base_pop_directory = Path(config["base_data"]["output_directory"])
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


# %%
def forecast_population_for_gor(
    config: dict, base_year: int, forecast_year: int, gor: str
):
    output_targets = config["output_targets"]

    # --- Step 0 --- #
    # Importing data
    LOGGER.info("--- Step 0 ---")
    # Read in the data
    LOGGER.info("Reading in the forecasting data for population forecasts")
    LOGGER.info(f"For {gor}, base year {base_year}, forecast year {forecast_year}.")

    pop_growth_factor = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="pop_growth_factor",
        geography_subset=gor,
        hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
    )

    soc_splits_change = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="soc_splits_change",
        geography_subset=gor,
        hdf_key=f"change_from_{base_year}_to_{forecast_year}",
    )

    base_pop_directory = Path(config["base_data"]["output_directory"])
    base_pop_file_stem = config["base_data"]["base_pop_file_stem"]
    filepath = base_pop_directory / f"{base_pop_file_stem}_{gor}.hdf"

    LOGGER.info(f"Loading base population file from {filepath}")
    base_pop = DVector.load(filepath)

    pop_g_adults_growth_factors = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="pop_single_household_adult_growth_factor",
        geography_subset=gor,
        hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
    )

    # --- Step 1 --- #
    # Prepare base files into forecasting segmentations
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Prepare base files into forecasting segmentations")

    base_pop_ntem_age = base_pop.translate_segment(
        from_seg="age_9", to_seg="age_ntem", drop_from=True
    )

    base_pop_age_ntem_g = base_pop_ntem_age.aggregate(segs=["age_ntem", "g", "soc"])

    base_pop_age_ntem_g_gor = base_pop_age_ntem_g.translate_zoning(
        new_zoning=pop_growth_factor.zoning_system,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    # --- Step 2 --- #
    # Calculate the IPF targets
    LOGGER.info("--- Step 2 ---")
    LOGGER.info("Calculate the IPF targets")

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
    base_pop_g_adults = base_pop.aggregate(segs=["g", "adults"]).filter_segment_value(
        "adults", [1]
    )

    base_pop_g_adults_lad19 = base_pop_g_adults.translate_zoning(
        new_zoning=pop_g_adults_growth_factors.zoning_system,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    pop_g_adults_targets = base_pop_g_adults_lad19 * pop_g_adults_growth_factors

    # and the lad targets need to be converted to a compatible 2021 zone system, being a combination of lsoa 2021 zones
    pop_g_adults_targets = pop_g_adults_targets.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f"LAD2021-{gor}"),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    # --- Step 3 --- #
    # Calculate the new SOC splits
    LOGGER.info("--- Step 3 ---")
    LOGGER.info("Calculate the new SOC splits")

    base_pop_gor = base_pop.translate_zoning(pop_growth_factor.zoning_system)

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

    # --- Step 4 --- #
    # Now apply the IPF using age_ntem, g, soc and adults
    LOGGER.info("--- Step 4 ---")
    LOGGER.info("Apply the IPF")
    rebalanced_pop, summary, differences = data_processing.apply_ipf(
        seed_data=base_pop_ntem_age,
        target_dvectors=[pop_targets, soc_targets, pop_g_adults_targets],
        cache_folder=constants.CACHE_FOLDER,
    )
    output_reference = f"Population_age_g_soc_{gor}_{forecast_year}"
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=output_reference,
        dvector=rebalanced_pop,
        dvector_dimension="people",
        output_level=OutputLevel.INTERMEDIATE,
    )

    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{output_reference}_VALIDATION.csv",
        float_format="%.5f",
        index=False,
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE,
        file=f"{output_reference}_VALIDATION.xlsx",
        dfs=differences,
    )

    seg_totals = data_processing.find_segment_totals(
        dvec=rebalanced_pop, dimension="population"
    )
    seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE
        / f"{output_reference}_segment_totals.csv",
        float_format="%.5f",
        index=False,
    )

    if output_targets:
        data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=f"pop_targets_{gor}_{forecast_year}",
            dvector=pop_targets,
            dvector_dimension="people",
            output_level=OutputLevel.INTERMEDIATE,
        )

        data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=f"soc_targets_{gor}_{forecast_year}",
            dvector=soc_targets,
            dvector_dimension="people",
            output_level=OutputLevel.INTERMEDIATE,
        )

        data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=f"pop_g_adults_targets_{gor}_{forecast_year}",
            dvector=pop_g_adults_targets,
            dvector_dimension="people",
            output_level=OutputLevel.INTERMEDIATE,
        )


def process_households_for_gor(
        config: dict, base_year: int, forecast_year: int, gor: str
):
    output_targets = config["output_targets"]
    # --- Step 0 --- #
    LOGGER.info("--- Step 0 ---")
    # Read in the data
    LOGGER.info("Reading in the forecasting data (for household forecasts)")
    LOGGER.info(f"For {gor}, base year {base_year}, forecast year {forecast_year}.")

    totals_growth_factor = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="pop_household_growth_factors",
        geography_subset=gor,
        hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
    )

    children_growth_factor = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="pop_household_children_growth_factors",
        geography_subset=gor,
        hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
    )

    single_adults_growth_factors = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="pop_single_household_adult_growth_factor_no_g",
        geography_subset=gor,
        hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
    )

    base_pop_directory = Path(config["base_data"]["output_directory"])
    base_hh_file_stem = config["base_data"]["base_households_file_stem"]
    filepath = base_pop_directory / f"{base_hh_file_stem}_{gor}.hdf"
    LOGGER.info(f"Loading file from {filepath}")
    base_hhs = DVector.load(filepath)

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Calculate the households growth targets")

    if "total" in base_hhs.segmentation.names:
        base_hhs_totals = base_hhs.aggregate(segs=["total"])
    else:
        base_hhs_totals = base_hhs.add_segments(new_segs=["total"]).aggregate(
            segs=["total"]
        )

    base_hhs_children = base_hhs.aggregate(segs=["children"])

    base_hhs_totals_gor = base_hhs_totals.translate_zoning(
        new_zoning=totals_growth_factor.zoning_system,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    base_hhs_children_gor = base_hhs_children.translate_zoning(
        new_zoning=children_growth_factor.zoning_system,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    hh_totals_targets = base_hhs_totals_gor * totals_growth_factor

    hh_children_targets = base_hhs_children_gor * children_growth_factor

    base_hh_single_adults = base_hhs.aggregate(segs=["adults"]).filter_segment_value(
        "adults", [1]
    )

    base_hh_single_adults_lad19 = base_hh_single_adults.translate_zoning(
        new_zoning=single_adults_growth_factors.zoning_system,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    hh_single_adults_targets = base_hh_single_adults_lad19 * single_adults_growth_factors

    # and the lad targets need to be converted to a compatible 2021 zone system, being a combination of lsoa 2021 zones
    hh_adults_targets = hh_single_adults_targets.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f"LAD2021-{gor}"),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")
    # Apply the IPF to targets based on children and total households
    rebalanced_hhs, summary, differences = data_processing.apply_ipf(
        seed_data=base_hhs,
        target_dvectors=[hh_children_targets, hh_adults_targets, hh_totals_targets],
        cache_folder=constants.CACHE_FOLDER,
    )
    output_reference = f"Households_{gor}_{forecast_year}"
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=output_reference,
        dvector=rebalanced_hhs,
        dvector_dimension="people",
        output_level=OutputLevel.INTERMEDIATE,
    )

    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{output_reference}_VALIDATION.csv",
        float_format="%.5f",
        index=False,
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE,
        file=f"{output_reference}_VALIDATION.xlsx",
        dfs=differences,
    )

    seg_totals = data_processing.find_segment_totals(
        dvec=rebalanced_hhs, dimension="households"
    )
    seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE
        / f"{output_reference}_segment_totals.csv",
        float_format="%.5f",
        index=False,
    )

    if output_targets:
        data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=f"hh_totals_targets_{gor}_{forecast_year}",
            dvector=hh_totals_targets,
            dvector_dimension="households",
            output_level=OutputLevel.INTERMEDIATE,
        )

        data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=f"hh_children_targets_{gor}_{forecast_year}",
            dvector=hh_children_targets,
            dvector_dimension="households",
            output_level=OutputLevel.INTERMEDIATE,
        )

        data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=f"hh_adults_targets_{gor}_{forecast_year}",
            dvector=hh_adults_targets,
            dvector_dimension="households",
            output_level=OutputLevel.INTERMEDIATE,
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

# for region in run_for_regions:
#     base_pop = fetch_base_pop(config=config, gor=region)

for forecast_year in forecast_years:
    for region in run_for_regions:
        forecast_population_for_gor(
            config=config, base_year=base_year, forecast_year=forecast_year, gor=region
        )

        process_households_for_gor(
            config=config, base_year=base_year, forecast_year=forecast_year, gor=region
        )
