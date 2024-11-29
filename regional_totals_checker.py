from argparse import ArgumentParser
from pathlib import Path

import yaml

from caf.base.zoning import TranslationWeighting
from caf.base.data_structures import DVector

from land_use import constants, data_processing


# TODO: expand on the documentation here
parser = ArgumentParser('Land-Use base employment command line runner')
parser.add_argument('config_file', type=Path)
args = parser.parse_args()

# load configuration file
with open(args.config_file, 'r') as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

block = 'base_data'

lad_raw_4_digit_sic = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='lad_raw_4_digit_sic'
)
# if warning is made about only being 350 columns after filter then this is okay as Northern Ireland is not in shapefile

msoa_raw_2_digit_sic = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='msoa_raw_2_digit_sic'
)

lsoa_raw_1_digit_sic = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='lsoa_raw_1_digit_sic'
)

ons_sic_soc_jobs_lu = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='ons_sic_soc_jobs_lu'
)

wfj = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='wfj'
)

soc_4_factors = data_processing.read_dvector_from_config(
    config=config,
    data_block=block,
    key='soc_4_factors'
)


def find_regional_totals(dvec:DVector):
    try:
        totals = dvec.add_segments(
            [constants.CUSTOM_SEGMENTS["total"]], split_method="split"
            ).aggregate(["total"])
    except ValueError:
        totals = dvec

    totals_at_rgn = totals.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    print(totals_at_rgn.data)

find_regional_totals(dvec=lad_raw_4_digit_sic)
find_regional_totals(dvec=msoa_raw_2_digit_sic)
find_regional_totals(dvec=lsoa_raw_1_digit_sic)
find_regional_totals(dvec=ons_sic_soc_jobs_lu)
find_regional_totals(dvec=wfj)
