from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml

# import caf.core as cc
# from caf.core.data_structures import DVector
# from caf.core.segments import SegmentsSuper
# from caf.core.zoning import TranslationWeighting

from land_use import constants, data_processing
from land_use import logging as lu_logging


# TODO: expand on the documentation here
# parser = ArgumentParser('Land-Use base employment command line runner')
# parser.add_argument('config_file', type=Path)
# args = parser.parse_args()



# load configuration file
# with open(args.config_file, 'r') as text_file:
#     config = yaml.load(text_file, yaml.SafeLoader)
config_file = Path(r"scenario_configurations\iteration_5\base_employment_hse_config.yml")
with open(config_file, 'r') as text_file:
     config = yaml.load(text_file, yaml.SafeLoader)

# Get output directory for intermediate outputs from config file
OUTPUT_DIR = Path(config["output_directory"])
OUTPUT_DIR.mkdir(exist_ok=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = bool(config["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(OUTPUT_DIR, log_name='employment')

# --- Step 0 --- #
# read in the data from the config file
block = 'base_data'
LOGGER.info("Importing data from config file")

# lad_raw_4_digit_sic = data_processing.read_dvector_from_config(
#     config=config,
#     data_block=block,
#     key='lad_raw_4_digit_sic'
# )

msoa_raw_2_digit_sic = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='msoa_raw_2_digit_sic'
)

# lsoa_raw_1_digit_sic = data_processing.read_dvector_from_config(
#     config=config,
#     data_block=block,
#     key='lsoa_raw_1_digit_sic'
# )

