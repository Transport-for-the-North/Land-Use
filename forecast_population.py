# %%
from pathlib import Path

import pandas as pd

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel

# %%
ons_pop_forecast_dir = Path(
    r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs\preprocessing"
)
ons_hh_forecast_dir = Path(
    r"I:\NorMITs Land Use\2023\import\ONS\forecasting\hh_projs\preprocessing"
)

base_year = 2023


# %%
base_pop_dir = Path(r"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs")
soc_dir = Path(
    r"I:\NorMITs Land Use\2023\import\Labour Market and Skills\LMS_SOC\preprocessing"
)

OUTPUT_DIR = Path(r"F:\Working\Land-Use\temp_forecast_population_testing")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
# generate_summary_outputs = True  # bool(config["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="population"
)


# %%


def process_region(gor: str, forecast_year: int, output_targets: bool):

    # --- Step 0 --- #
    LOGGER.info("--- Step 0 ---")
    # read in the currently hard coded but switch to config

    LOGGER.info(f"Importing data for {gor}")  # eventually from the config file

    geographical_level, geographical_subset = fetch_gor_info(gor=gor)

    pop_growth_factor = data_processing.read_dvector_data(
        file_path=ons_pop_forecast_dir
        / f"pop_growth_factors_{base_year}_to_{forecast_year}.hdf",
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )

    soc_splits_change_path = soc_dir / f"soc_pp_change_{base_year}_to_{forecast_year}.hdf"

    soc_splits_change = data_processing.read_dvector_data(
        file_path=soc_splits_change_path,
        geographical_level=geographical_level,
        input_segments=["g", "soc"],
        geography_subset=geographical_subset,
    )

    hh_1_adults_g_base_path = ons_hh_forecast_dir / f'hh_1_adult_by_g_{base_year}.hdf'
    hh_1_adults_g_forecast_path = ons_hh_forecast_dir / f'hh_1_adult_by_g_{forecast_year}.hdf'
    hh_1_adults_g_base = data_processing.read_dvector_data(
        file_path=hh_1_adults_g_base_path,
        geographical_level="LAD2019_EWS",
        input_segments=["g", "adults"],
        geography_subset=None,  # geographical subset?
    )
    hh_1_adults_g_forecast = data_processing.read_dvector_data(
        file_path=hh_1_adults_g_forecast_path,
        geographical_level="LAD2019_EWS",
        input_segments=["g", "adults"],
        geography_subset=None,
    )

    base_pop = DVector.load(base_pop_dir / f"Output P11_{gor}.hdf")

    # --- Step 1 --- #
    # Prepare base files into forecasting segmentations
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Prepare base files into forecasting segmentations")

    base_pop_ntem_age = base_pop.translate_segment(
        from_seg="age_9", to_seg="age_ntem", drop_from=True
    )

    base_pop_age_ntem_g = base_pop_ntem_age.aggregate(segs=["age_ntem", "g", "soc"])

    base_pop_age_ntem_g_gor = base_pop_age_ntem_g.translate_zoning(
        new_zoning=pop_growth_factor.zoning_system,  # fix zoning system to match
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
    base_pop_g_adults = base_pop.aggregate(segs=["g", "adults"]).filter_segment_value("adults", [1])

    pop_g_adults_growth_factors = hh_1_adults_g_forecast / hh_1_adults_g_base

    base_pop_g_adults_lad19 = base_pop_g_adults.translate_zoning(
        new_zoning=pop_g_adults_growth_factors.zoning_system,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT
    )

    pop_g_adults_targets = base_pop_g_adults_lad19 * pop_g_adults_growth_factors  # in LAD19 zoning

    # --- Step 3 --- #
    # Calculate the new SOC splits
    LOGGER.info("--- Step 3 ---")
    LOGGER.info("Calculate the new SOC splits")

    p11_gor = base_pop.translate_zoning(pop_growth_factor.zoning_system)

    p11_gor_soc = p11_gor.aggregate(["g", "soc"]).filter_segment_value("soc", [1, 2, 3])
    p11_soc_totals = (
        p11_gor_soc.add_segments(["total"])
        .aggregate(["total"])
        .add_segments(["soc"])
        .filter_segment_value("soc", [1, 2, 3])
    )

    soc_p11_perc = p11_gor_soc / p11_soc_totals

    # work out the new targets splits
    soc_target_perc = soc_p11_perc + soc_splits_change
    # check for negative splits
    check_negatives(input_df=soc_target_perc.data)

    # the totals here should be the pop_targets without soc 4
    soc_targets = (soc_target_perc * base_pop_soc_exc_4_total).aggregate(["g", "soc"])
    # TODO change SOC to use the SOC by gender (at region level)

    # --- Step 4 --- #
    # Now apply the IPF using age_ntem, g, and soc
    # TODO add the 1 adult households to the IPF process
    LOGGER.info("--- Step 4 ---")
    LOGGER.info("Apply the IPF")
    rebalanced_pop, summary, differences = data_processing.apply_ipf(
        seed_data=base_pop_ntem_age,
        target_dvectors=[pop_targets, soc_targets],
        cache_folder=constants.CACHE_FOLDER,
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"Population_age_g_soc_{gor}_{forecast_year}",
        dvector=rebalanced_pop,
        dvector_dimension="people",
        output_level=OutputLevel.INTERMEDIATE,
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


def fetch_gor_info(gor: str) -> tuple[str, str | None]:

    if gor == "Scotland":
        return "SCOTLANDRGN", None
    return "RGN2021", gor


def check_negatives(input_df: pd.DataFrame):
    if (input_df < 0).any().any():
        raise ValueError(f"New SOC target splits calculated contain negatives")
    else:
        pass


def process_households(gor: str, forecast_year: int):
    # --- Step 0 --- #
    LOGGER.info("--- Step 0 ---")
    # Read in the data
    LOGGER.info("Reading in the forecasting data")

    if gor == "Scotland":
        geographical_level = "SCOTLANDRGN"
        geographical_subset = None
    else:
        geographical_level = "RGN2021"
        geographical_subset = gor

    regional_2018_base_year_totals = ons_hh_forecast_dir / f"hh_totals_{base_year}.hdf"

    regional_2018_forecast_year_totals = (
            ons_hh_forecast_dir / f"hh_totals_{forecast_year}.hdf"
    )

    dv_base_year_totals = data_processing.read_dvector_data(
        file_path=regional_2018_base_year_totals,
        geographical_level=geographical_level,
        input_segments=["total"],
        geography_subset=geographical_subset,
    )

    dv_forecast_year_totals = data_processing.read_dvector_data(
        file_path=regional_2018_forecast_year_totals,
        geographical_level=geographical_level,
        input_segments=["total"],
        geography_subset=geographical_subset,
    )

    regional_2018_base_year_children = ons_hh_forecast_dir / f"hh_children_{base_year}.hdf"

    regional_2018_forecast_year_children = (
        ons_hh_forecast_dir / f"hh_children_{forecast_year}.hdf"
    )

    dv_base_year_children = data_processing.read_dvector_data(
        file_path=regional_2018_base_year_children,
        geographical_level=geographical_level,
        input_segments=["children"],
        geography_subset=geographical_subset,
    )

    dv_forecast_year_children = data_processing.read_dvector_data(
        file_path=regional_2018_forecast_year_children,
        geographical_level=geographical_level,
        input_segments=["children"],
        geography_subset=geographical_subset,
    )

    filepath = base_pop_dir / f"Output P13.3_{gor}.hdf"
    base_hhs = DVector.load(filepath)

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Calculate the households growth targets")

    totals_growth_factor = dv_forecast_year_totals / dv_base_year_totals

    children_growth_factor = dv_forecast_year_children / dv_base_year_children

    base_hhs_totals = base_hhs.aggregate(segs=["total"])

    base_hhs_children = base_hhs.aggregate(segs=["children"])

    base_hhs_totals_gor = base_hhs_totals.translate_zoning(
        new_zoning=totals_growth_factor.zoning_system,  # fix zoning system to match growth factors
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    base_hhs_children_gor = base_hhs_children.translate_zoning(
        new_zoning=children_growth_factor.zoning_system,  # fix zoning system to match growth factors
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    hh_totals_targets = base_hhs_totals_gor * totals_growth_factor

    hh_children_targets = base_hhs_children_gor * children_growth_factor

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"hh_totals_targets_{forecast_year}_{gor}",
        dvector=hh_totals_targets,
        dvector_dimension="households",
        output_level=OutputLevel.INTERMEDIATE,
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"hh_children_targets_{forecast_year}_{gor}",
        dvector=hh_children_targets,
        dvector_dimension="households",
        output_level=OutputLevel.INTERMEDIATE,
    )

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")
    # Apply the IPF to targets based on children and total households
    rebalanced_hhs, summary, differences = data_processing.apply_ipf(
        seed_data=base_hhs,
        target_dvectors=[hh_children_targets, hh_totals_targets],
        cache_folder=constants.CACHE_FOLDER,
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"Output p11_age_g_soc_children_{gor}_{forecast_year}",
        dvector=rebalanced_hhs,
        dvector_dimension="people",
        output_level=OutputLevel.INTERMEDIATE,
    )

# # takes a while to run. So suggest this is run only when needed
# for gor in constants.GORS + ["Scotland"]:
#     print(gor)
#     process_region(gor=gor, forecast_year=2038)

# for gor in constants.GORS + ["Scotland"]:
#     print(gor)
#     process_region(gor=gor, forecast_year=2048)

# testing as quicker than looping through all regions

regions = [
    "NW",
]
forecast_years = [
    2043,
]

for region in regions:
    for forecast_year in forecast_years:
        process_region(gor=region, forecast_year=forecast_year, output_targets=True)
