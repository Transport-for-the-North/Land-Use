# -*- coding: utf-8 -*-
"""
    Main module for ABP processing, to extract data from the database
    and aggregate to zone system.
"""

##### IMPORTS #####
# Standard imports
import datetime as dt
import logging
import pathlib

# Third party imports

# Local imports
from land_use.abp_processing import abp_processing, config


##### CONSTANTS #####
LOG = logging.getLogger(__package__)
CONFIG_FILE = pathlib.Path("abp_processing_config.yml")
LOG_FILE = "ABP_processing.log"


##### FUNCTIONS #####
def main():
    """Main entry point function in ABP processing.

    Loads config (`CONFIG_FILE`) and runs ABP processing
    `extract_data` function.
    """
    parameters = config.ABPExtractConfig.load_yaml(CONFIG_FILE)
    output_folder = parameters.output_folder / f"{dt.date.today():%Y%m%d} ABP Extract"
    output_folder.mkdir(exist_ok=True)

    abp_processing.initialise_logger(output_folder / LOG_FILE)
    LOG.info("Loaded parameters from config: %s", CONFIG_FILE)
    LOG.info("Outputs saved to: %s", output_folder)

    abp_processing.extract_data(
        parameters.database_connection_parameters,
        output_folder,
        parameters.filter_codes,
        parameters.year,
        parameters.aggregate_shapefile,
    )


if __name__ == "__main__":
    main()
