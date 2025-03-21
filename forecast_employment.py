# %%
from pathlib import Path

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

OUTPUT_DIR = Path(r"F:\Working\Land-Use\OUTPUTS_forecast_employment")
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
# TODO load configuration file and use "read_dvector_from_config")


def process_forecast_emp(forecast_year: int):
    LOGGER = lu_logging.configure_logger(
        OUTPUT_DIR / OutputLevel.SUPPORTING, log_name=f"employment_{forecast_year}"
    )

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

    # Load in the Base employment output
    base_emp = DVector.load(base_emp_dir / "Output E6.hdf")

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    # Prepare the base files into forecasting segmentation and calculate the growth targets
    LOGGER.info("Prepare base files into forecasting segmentations and calculate growth targets")

    # Translate Base Emp DVector into region zoning (England, Scotland, Wales)
    base_emp_rgn = base_emp.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )

    growth_factors = lms_sic_forecast / lms_sic_base

    base_emp_agg = base_emp_rgn.aggregate(segs=["sic_1_digit"])

    sic_1_digit_targets = base_emp_agg * growth_factors
    # Drop targets for SIC level -1, 20, 21 (potentially move this to the reformatting script)
    sic_1_digit_targets = drop_seg_values(sic_1_digit_targets, "sic_1_digit", [-1, 20, 21])

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"sic_1_digit_targets_{forecast_year}",
        dvector=sic_1_digit_targets,
        dvector_dimension="jobs",
        output_level=OutputLevel.INTERMEDIATE,
    )

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")
    # Apply the IPF to targets based on SIC 1 digit
    LOGGER.info("Apply the IPF")
    rebalanced_emp, summary, differences = data_processing.apply_ipf(
        seed_data=base_emp,
        target_dvectors=[sic_1_digit_targets],
        cache_folder=constants.CACHE_FOLDER
    )

    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f"Output Emp_SIC_1_digit_{forecast_year}",
        dvector=rebalanced_emp,
        dvector_dimension="jobs",
        output_level=OutputLevel.INTERMEDIATE,
    )


# --- Useful function that probably should be DVector method --- #
def drop_seg_values(dvec: DVector, segment_name:str, drop_values: list[int]) -> DVector:
    """Drop rows with provided seg values for the given segmentation, keep other rows
    Args:
        dvec (DVector): DVector function will be applied to
        segment_name (str): The name of the segment to filter by.
        drop_values (list[int]): Segment values to drop
    Returns:
        DVector: Dvector with values removed for the given segment
    """

    current_values = dvec.data.index.get_level_values(segment_name).tolist()
    keep_values = list(set(current_values) - set(drop_values))
    return dvec.filter_segment_value(segment_name, keep_values)


process_forecast_emp(forecast_year=2033)
process_forecast_emp(forecast_year=2038)
process_forecast_emp(forecast_year=2043)
process_forecast_emp(forecast_year=2048)
process_forecast_emp(forecast_year=2053)
