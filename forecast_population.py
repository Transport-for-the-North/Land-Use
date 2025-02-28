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

base_year = 2023
forecast_year = 2043

# if beyond 2043 then need to get
# the regional ones at 2043,
# the 2018 national for 2043
# the 2022 national forecast for 2048
# first growth from 2023 to 2043 (2022),
# and then growth from 2043 to 2048 using 2022

# %%
base_emp_dir = Path(r"F:\Deliverables\Land-Use\241213_Population\02_Final Outputs")
soc_dir = Path(
    r"I:\NorMITs Land Use\2023\import\Labour Market and Skills\LMS_SOC\preprocessing"
)

OUTPUT_DIR = Path(r"F:\Working\Land-Use\OUTPUTS_forecast_population")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = True  # bool(config["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="employment"
)


# %%


def process_region(gor: str):
    # --- Step 0 --- #
    # read in the currently hard coded but switch to config

    LOGGER.info(f"Importing data for {gor}")  # eventually from the config file

    if gor == "Scotland":
        geographical_level = "SCOTLANDRGN"
        geographical_subset = None
    else:
        geographical_level = "RGN2021"
        geographical_subset = gor

    national_2021_22_base = (
        ons_pop_forecast_dir / f"2021_22_based_ews_pop_projections_{base_year}.hdf"
    )
    national_2021_22_forecast = (
        ons_pop_forecast_dir / f"2021_22_based_ews_pop_projections_{forecast_year}.hdf"
    )

    dv_national_2022_base = data_processing.read_dvector_data(
        file_path=national_2021_22_base,
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )

    dv_national_2022_forecast = data_processing.read_dvector_data(
        file_path=national_2021_22_forecast,
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )

    regional_2021_22_base = (
        ons_pop_forecast_dir / f"2018_20_21_regions_pop_projections_{base_year}.hdf"
    )
    regional_2021_22_forecast = (
        ons_pop_forecast_dir / f"2018_20_21_regions_pop_projections_{forecast_year}.hdf"
    )

    dv_regional_base = data_processing.read_dvector_data(
        file_path=regional_2021_22_base,
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )

    dv_regional_forecast = data_processing.read_dvector_data(
        file_path=regional_2021_22_forecast,
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )
    national_2018_20_21_base = (
        ons_pop_forecast_dir / f"2018_20_21_country_pop_projections_{base_year}.hdf"
    )
    national_2018_20_21_forecast = (
        ons_pop_forecast_dir / f"2018_20_21_country_pop_projections_{forecast_year}.hdf"
    )

    dv_national_2018_base = data_processing.read_dvector_data(
        file_path=national_2018_20_21_base,
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )

    dv_national_2018_forecast = data_processing.read_dvector_data(
        file_path=national_2018_20_21_forecast,
        geographical_level=geographical_level,
        input_segments=["age_ntem", "g"],
        geography_subset=geographical_subset,
    )

    soc_base_path = soc_dir / f"LMS_SOC_Occ_T1_{base_year}.hdf"

    soc_base = data_processing.read_dvector_data(
        file_path=soc_base_path,
        geographical_level="RGN2021",
        input_segments=["soc"],
        geography_subset=gor,
    )

    soc_forecast_path = soc_dir / f"LMS_SOC_Occ_T1_{forecast_year}.hdf"

    soc_forecast = data_processing.read_dvector_data(
        file_path=soc_forecast_path,
        geographical_level="RGN2021",
        input_segments=["soc"],
        geography_subset=gor,
    )

    filepath = base_emp_dir / f"Output P11_{gor}.hdf"
    p11 = DVector.load(filepath)

    # --- Step 1 --- #
    # Prepare base files into forecasting segmentations
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("Prepare base files into forecasting segmentations")

    # switch p11's age segmentation from age_9 to age_ntem
    p11_ntem_age = p11.add_segments(new_segs=["age_ntem"])

    p11_ntem_age = p11_ntem_age.aggregate(
        segs=[seg for seg in p11_ntem_age.data.index.names if seg != "age_9"]
    )

    # --- Step 2 --- #
    # Calculate the population growth factors
    LOGGER.info("--- Step 2 ---")
    LOGGER.info("Calculate the population growth factors")

    # forecast corrections
    uplift_base_year_factor = dv_national_2022_base / dv_national_2018_base
    uplift_forecast_year_factor = dv_national_2022_forecast / dv_national_2018_forecast
    # Adjust for 2022

    adj_base_year = uplift_base_year_factor * dv_regional_base
    adj_future_year = uplift_forecast_year_factor * dv_regional_forecast

    adj_growth_factor = adj_future_year / adj_base_year

    p11_age_ntem_g = p11_ntem_age.aggregate(segs=["age_ntem", "g"])

    p11_age_ntem_g_gor = p11_age_ntem_g.translate_zoning(
        new_zoning=adj_growth_factor.zoning_system,  # fix zoning system to match growth factors
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    pop_targets = p11_age_ntem_g_gor * adj_growth_factor

    # --- Step 3 --- #
    # Apply the IPF to targets based on age and gender

    rebalanced_p11, summary, differences = data_processing.apply_ipf(
        seed_data=p11_ntem_age,
        target_dvectors=[pop_targets],
        cache_folder=constants.CACHE_FOLDER,
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"Output p11_ge_g_{gor}",
        dvector=rebalanced_p11,
        dvector_dimension="people",
        output_level=OutputLevel.INTERMEDIATE,
    )

    # --- Step 4 --- #
    # Calculate population numbers (post initial IPF) excluding soc 4

    rebalanced_p11_gor = rebalanced_p11.translate_zoning(soc_base.zoning_system)

    rebalanced_p11_soc_totals = rebalanced_p11_gor.aggregate(["soc"])

    rebalanced_p11_soc_totals_exc_4 = rebalanced_p11_soc_totals.filter_segment_value(
        "soc", [1, 2, 3]
    )

    rebalanced_p11_soc_totals_exc_4_totals = (
        rebalanced_p11_soc_totals_exc_4.add_segments(["total"]).aggregate(["total"])
    )

    rebalanced_p11_soc_totals_exc_4_totals = (
        rebalanced_p11_soc_totals_exc_4_totals.add_segments(
            ["soc"]
        ).filter_segment_value("soc", [1, 2, 3])
    )

    # --- Step 5 --- #
    # Calculate the new SOC splits

    # soc 4 is excluded throughout as do not have a forecast for this segment
    soc_base_totals = (
        soc_base.add_segments(["total"])
        .aggregate(["total"])
        .add_segments(["soc"])
        .filter_segment_value("soc", [1, 2, 3])
    )

    soc_forecast_totals = (
        soc_forecast.add_segments(["total"])
        .aggregate(["total"])
        .add_segments(["soc"])
        .filter_segment_value("soc", [1, 2, 3])
    )

    # calcate the soc splits from the forecast data
    soc_base_perc = soc_base / soc_base_totals
    soc_forecast_perc = soc_forecast / soc_forecast_totals

    # work out the change in per splits
    soc_splits_change = soc_forecast_perc - soc_base_perc

    p11_gor = p11.translate_zoning(soc_base.zoning_system)

    p11_gor_soc = p11_gor.aggregate(["soc"]).filter_segment_value("soc", [1, 2, 3])
    p11_soc_totals = (
        p11_gor_soc.add_segments(["total"])
        .aggregate(["total"])
        .add_segments(["soc"])
        .filter_segment_value("soc", [1, 2, 3])
    )

    soc_p11_perc = p11_gor_soc / p11_soc_totals

    # work out the new targets splits 
    # TODO: add in check to make sure this isn't negative, and if so need to do something about it.
    soc_target_perc = soc_p11_perc + soc_splits_change

    # the totals here should be the pop_targets without soc 4
    soc_targets = (soc_target_perc * rebalanced_p11_soc_totals_exc_4_totals).aggregate(
        ["soc"]
    )

    # --- Step 6 --- #
    # Apply the IPF to targets based on age, gender and soc

    rebalanced_age_g_soc_p11, summary, differences = data_processing.apply_ipf(
        seed_data=p11_ntem_age,
        target_dvectors=[pop_targets, soc_targets],
        cache_folder=constants.CACHE_FOLDER,
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"Output p11_age_g_soc_{gor}",
        dvector=rebalanced_age_g_soc_p11,
        dvector_dimension="people",
        output_level=OutputLevel.INTERMEDIATE,
    )


# # takes a while to run. So suggest this is run only when needed
# for gor in constants.GORS:
#     print(gor)
#     process_region(gor=gor)

# # and Scotland
# print("scotland")
# process_region(gor="Scotland")

# testing NW as quicker than looping through all regions
process_region(gor="NW")
