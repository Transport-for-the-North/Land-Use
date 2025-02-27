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

OUTPUT_DIR = Path(r"F:\Working\Land-Use\OUTPUTS_forecast_population")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = True #bool(config["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(OUTPUT_DIR / OutputLevel.SUPPORTING, log_name='employment')


# %%


def process_region(gor: str):
    read_in_files(gor=gor)


def read_in_files(gor: str):
    # --- Step 0 --- #
    # read in the currently hard coded but switch to config

    LOGGER.info("Importing data ") # eventually from the config file

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

    filepath = base_emp_dir / f"Output P11_{gor}.hdf"
    p11 = DVector.load(filepath)

    # --- Step 1 --- #
    # Calculate the population growth factors
    LOGGER.info('--- Step 1 ---')
    LOGGER.info('Calculate the population growth factors')

    # forecast corrections
    uplift_base_year_factor = dv_national_2022_base / dv_national_2018_base
    uplift_forecast_year_factor = dv_national_2022_forecast / dv_national_2018_forecast
    # Adjust for 2022

    adj_base_year = uplift_base_year_factor * dv_regional_base
    adj_future_year = uplift_forecast_year_factor * dv_regional_forecast

    adj_growth_factor = adj_future_year / adj_base_year

    p11 = p11.add_segments(new_segs=["age_ntem"])
    p11_age_ntem_g = p11.aggregate(segs=["age_ntem", "g"])

    p11_age_ntem_g_gor = p11_age_ntem_g.translate_zoning(
        new_zoning=adj_growth_factor.zoning_system,  # fix zoning system to match growth factors
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
    )

    pop_targets = p11_age_ntem_g_gor * adj_growth_factor

    rebalanced_p11, summary, differences = data_processing.apply_ipf(
        seed_data=p11_age_ntem_g,
        target_dvectors=[pop_targets],
        cache_folder=constants.CACHE_FOLDER,
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output Pop_age_g_{gor}',
        dvector=rebalanced_p11,
        dvector_dimension='people',
        output_level=OutputLevel.INTERMEDIATE
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
