# %%
from pathlib import Path

import pandas as pd
import yaml

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel

# %%
base_emp_dir = Path(r"F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs")
sic_dir = Path(
    r"I:\NorMITs Land Use\2023\import\Labour Market and Skills\LMS_SIC_Ind2\preprocessing"
)

base_year = 2023
forecast_year = 2043

OUTPUT_DIR = Path(r"F:\Working\Land-Use\OUTPUTS_forecast_employment")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = True

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="employment"
)

# %%
# TODO load configuration file and use "read_dvector_from_config")


def process_forecast_emp():
    # --- Step 0 --- #
    LOGGER.info("--- Step 0 ---")
    # Read in the data
    LOGGER.info("Reading in the forecasting data")
    # LOGGER.info(f"Importing data for {gor}")

    geographical_level = "RGN2021+SCOTLANDRGN"

    # Load in LM&S totals as DVector
    lms_sic_base_file = sic_dir / fr"LMS_SIC_1_digit_Ind2_{base_year}.hdf"

    lms_sic_base = data_processing.read_dvector_data(
        file_path=lms_sic_base_file,
        geographical_level=geographical_level,
        input_segments=["sic_1_digit"]
    )

    lms_sic_forecast_file = sic_dir / fr"LMS_SIC_1_digit_Ind2_{forecast_year}.hdf"

    lms_sic_forecast = data_processing.read_dvector_data(
        file_path=lms_sic_forecast_file,
        geographical_level=geographical_level,
        input_segments=["sic_1_digit"]
    )

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    # Prepare the base files into forecasting segmentation
    LOGGER.info("Prepare base files into forecasting segmentations")

    output_e6 = DVector.load(base_emp_dir / "Output E6.hdf")

    # Translate Output E6 DVector into region zoning (England, Scotland, Wales)
    output_e6_rgn = output_e6.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")
    # Calculate the growth factors and targets
    LOGGER.info("Calculate the growth factors")

    growth_factors = lms_sic_forecast / lms_sic_base

    output_e6_agg = output_e6_rgn.aggregate(segs=["sic_1_digit"])

    sic_1_digit_targets = output_e6_agg * growth_factors
    # Drop targets for SIC level -1, 20, 21 (potentially move this to the reformatting script)
    sic_1_digit_targets = sic_1_digit_targets.filter_segment_value(
        "sic_1_digit", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    # --- Step 3 --- #
    LOGGER.info("--- Step 3 ---")
    # Apply the IPF to targets based on SIC 1 digit
    rebalanced_e6, summary, differences = data_processing.apply_ipf(
        seed_data=output_e6,
        target_dvectors=[sic_1_digit_targets],
        cache_folder=constants.CACHE_FOLDER
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"Output E6_IPF_SIC_1_digit",
        dvector=rebalanced_e6,
        dvector_dimension="jobs",
        output_level=OutputLevel.INTERMEDIATE,
    )
