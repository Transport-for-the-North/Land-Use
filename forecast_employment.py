# %%
from argparse import ArgumentParser
from pathlib import Path
import shutil

import yaml

from caf.base.data_structures import DVector

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel


def process_forecast_emp(config: dict, forecast_year: int) -> None:

    # --- Step 0 --- #
    # Load in the Base employment output
    base_emp_path = Path(config["base_data"]["base_emp_filepath"])
    base_emp = DVector.load(base_emp_path)

    base_seg_totals = data_processing.find_segment_totals(
        dvec=base_emp, dimension="employment"
    )
    base_seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE
        / "emp_base_segment_totals.csv",
        float_format="%.5f",
        index=False,
    )

    # load in sic targets
    sic_targets = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="sic_targets",
        hdf_key=f"targets_{forecast_year}",
    )

    # load in soc targets
    soc_targets = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="soc_targets",
        hdf_key=f"targets_{forecast_year}",
    )

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")

    # Apply the IPF to targets
    LOGGER.info("Apply the IPF to targets")
    rebalanced_emp, summary, differences = data_processing.apply_ipf(
        seed_data=base_emp,
        target_dvectors=[sic_targets, soc_targets],
        cache_folder=constants.CACHE_FOLDER,
    )

    output_reference = f"Output Emp_sic_soc_{forecast_year}"
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

    forecast_seg_totals = data_processing.find_segment_totals(
        dvec=rebalanced_emp, dimension="employment"
    )
    forecast_seg_totals.to_csv(
        OUTPUT_DIR
        / OutputLevel.INTERMEDIATE
        / f"{output_reference}_segment_totals.csv",
        float_format="%.5f",
        index=False,
    )


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

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="employment"
)

# copy config file for traceability
shutil.copy(
    src=args.config_file,
    dst=OUTPUT_DIR / OutputLevel.SUPPORTING / args.config_file.name,
)

forecast_years = configuration["forecast_years"]

for forecast_year in forecast_years:
    process_forecast_emp(config=configuration, forecast_year=forecast_year)
