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


def fetch_base_emp(config: dict) -> DVector:
    # --- Step 0 --- #
    LOGGER.info("--- Step 0 ---")
    LOGGER.info("Load in the Base employment output")
    
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

    # and for the regional ones
    find_regional_seg_totals(dvec=base_emp, output_prefix="emp_base_segment_totals")

    return base_emp


def find_regional_seg_totals(dvec:DVector, output_prefix:str) -> None:

    # convert to region zone system (one column for each region)
    dvec_rgn = dvec.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    rgns = constants.GORS

    rgns.append("Scotland")

    # and find segment totals for each region
    for rgn in rgns:

        base_emp_one_rgn = dvec_rgn.translate_zoning(
            new_zoning=constants.KNOWN_GEOGRAPHIES[f"RGN2021-{rgn}"],
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False,
        )

        rgn_seg_totals = data_processing.find_segment_totals(
            dvec=base_emp_one_rgn, dimension="employment"
        )
        rgn_seg_totals.to_csv(
            OUTPUT_DIR
            / OutputLevel.INTERMEDIATE
            / f"{output_prefix}_{rgn}.csv",
            float_format="%.5f",
            index=False,
        )


def process_forecast_emp(config: dict, base_emp:DVector, forecast_year: int) -> None:

    # --- Step 1 --- #
    LOGGER.info("--- Step 1 ---")
    LOGGER.info("load in ipf targets")
    sic_targets = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="sic_targets",
        hdf_key=f"targets_{forecast_year}",
    )

    soc_targets = data_processing.read_dvector_from_config(
        config=config,
        data_block="forecast_data",
        key="soc_targets",
        hdf_key=f"targets_{forecast_year}",
    )

    # --- Step 2 --- #
    LOGGER.info("--- Step 2 ---")

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

    # and for the regional ones
    find_regional_seg_totals(dvec=base_emp, output_prefix=output_reference)


# %%
# load configuration file
# python forecast_employment.py "path/to_file/example_config.yml"

# TODO: expand on the documentation here
parser = ArgumentParser("Land-Use forecast employment command line runner")
parser.add_argument("config_file", type=Path)
args = parser.parse_args()

with open(args.config_file, "r") as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

# Get output directory for intermediate outputs from config file
OUTPUT_DIR = Path(config["output_directory"])
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

LOGGER = lu_logging.configure_logger(
    OUTPUT_DIR / OutputLevel.SUPPORTING, log_name="employment"
)

# copy config file for traceability
shutil.copy(
    src=args.config_file,
    dst=OUTPUT_DIR / OutputLevel.SUPPORTING / args.config_file.name,
)

forecast_years = config["forecast_years"]

base_emp = fetch_base_emp(config=config)

for forecast_year in forecast_years:
    process_forecast_emp(config=config, base_emp=base_emp, forecast_year=forecast_year)
