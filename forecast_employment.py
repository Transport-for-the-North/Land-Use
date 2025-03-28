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


def process_forecast_emp(config: dict, base_year: int, forecast_year: int) -> None:

    output_targets = config["output_targets"]

    # --- Step 0 --- #
    LOGGER.info("--- Step 0 ---")
    # Read in the data
    LOGGER.info("Reading in the forecasting data")

    sic_growth_factors = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="sic_employment_growth_factor",
        hdf_key=f"factors_from_{base_year}_to_{forecast_year}",
    )

    # Load in the Base employment output
    base_emp_path = Path(config["base_data"]["base_emp_filepath"])
    base_emp = DVector.load(base_emp_path)

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    # Prepare the base files into forecasting segmentation and calculate the growth targets
    LOGGER.info(
        "Prepare base files into forecasting segmentations and calculate growth targets"
    )

    # Translate Base Emp DVector into region zoning (England, Scotland, Wales)
    base_emp_rgn = base_emp.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    base_emp_agg = base_emp_rgn.aggregate(segs=["sic_1_digit"])

    sic_1_digit_targets = base_emp_agg * sic_growth_factors
    # Drop targets for SIC level -1, 20, 21 (potentially move this to the reformatting script)
    sic_1_digit_targets = drop_seg_values(
        sic_1_digit_targets, "sic_1_digit", [-1, 20, 21]
    )

    if output_targets:
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
        cache_folder=constants.CACHE_FOLDER,
    )

    output_reference = f"Output Emp_SIC_1_digit_{forecast_year}"
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=output_reference,
        dvector=rebalanced_emp,
        dvector_dimension="jobs",
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


# --- Useful function that probably should be DVector method --- #
def drop_seg_values(
    dvec: DVector, segment_name: str, drop_values: list[int]
) -> DVector:
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


# %%
# load configuration file
# python forecast_employment.py "path/to_file/example_config.yml"

# TODO: expand on the documentation here
parser = ArgumentParser("Land-Use forecast employment command line runner")
parser.add_argument("config_file", type=Path)
args = parser.parse_args()

with open(args.config_file, "r") as text_file:
    configuration = yaml.load(text_file, yaml.SafeLoader)

# Get output directory for intermediate outputs from config file
OUTPUT_DIR = Path(configuration["output_directory"])
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = bool(configuration["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="employment"
)

# copy config file for traceability
shutil.copy(
    src=args.config_file,
    dst=OUTPUT_DIR / OutputLevel.SUPPORTING / args.config_file.name,
)

base_year = configuration["base_year"]
forecast_years = configuration["forecast_years"]

for forecast_year in forecast_years:
    process_forecast_emp(
        config=configuration, base_year=base_year, forecast_year=forecast_year
    )
