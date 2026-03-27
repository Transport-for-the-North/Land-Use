from pathlib import Path

# from caf.base.segments import SegmentsSuper

import land_use.preprocessing as pp
from land_use.constants import geographies
import pandas as pd


# from argparse import ArgumentParser
from pathlib import Path

# import shutil

import pandas as pd
import yaml

import caf.base as cc
from caf.base.data_structures import DVector
from caf.base.segments import SegmentsSuper
from caf.base.zoning import TranslationWeighting

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel

INPUT_DIR = Path(r"I:\Data\NTEM\MG_NorCOM_review_121324")


def main():
    create_dv(year=2023)
    read_in_dv(year=2023)


def create_dv(year:int):
    filepath = INPUT_DIR / "ntem_8.0_Core_planning_data.csv"

    df = pd.read_csv(filepath)

    age_rows = ["under16", "16-74", "75+"]
    df = df[df["population"].isin(age_rows)]

    age_to_segment = {"under16": 1, "16-74": 2, "75+": 3}

    df["age_ntem"] = df["population"].map(age_to_segment)

    df_in_dvec_format = pp.pivot_to_dvector(
        data=df,
        zoning_column="msoa_zone_id",
        index_cols=["age_ntem"],
        value_column=str(year),
    )

    pp.save_preprocessed_hdf(
        source_file_path=filepath, 
        df=df_in_dvec_format, 
        multiple_output_ref=f"age_ntem_{year}"
    )


def read_in_dv(year:int):
    config_file = Path("ntem_8_age_ntem.yml")

    # load configuration file
    with open(config_file, "r") as text_file:
        config = yaml.load(text_file, yaml.SafeLoader)

    # Get output directory for intermediate outputs from config file
    OUTPUT_DIR = Path(config["output_directory"])
    OUTPUT_DIR.mkdir(exist_ok=True)

    block = "ntem_8"

    working_age = data_processing.read_dvector_from_config(
        config=config, data_block=block, key=f"age_ntem_{year}"
    )

    region_totals = working_age.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    file_stem = f"ntem 8 people by working age cat {2023}"

    # save output to hdf and csvs for checking
    data_processing.save_output(
            output_folder=OUTPUT_DIR,
            output_reference=file_stem,
            dvector=region_totals,
            dvector_dimension='population',
            output_level=OutputLevel.INTERMEDIATE
    )

    csv_filepath = OUTPUT_DIR / OutputLevel.INTERMEDIATE / f"{file_stem}.csv"
    region_totals.data.to_csv(csv_filepath)


if __name__ == "__main__":
    main()
