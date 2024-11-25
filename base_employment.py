from argparse import ArgumentParser
from pathlib import Path
import shutil

import pandas as pd
import yaml

import caf.base as cc
from caf.base.data_structures import DVector
from caf.base.segments import SegmentsSuper
from caf.base.zoning import TranslationWeighting

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel


# TODO: expand on the documentation here
parser = ArgumentParser('Land-Use base employment command line runner')
parser.add_argument('config_file', type=Path)
args = parser.parse_args()

# load configuration file
with open(args.config_file, 'r') as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)


# Get output directory for intermediate outputs from config file
OUTPUT_DIR = Path(config["output_directory"])
OUTPUT_DIR.mkdir(exist_ok=True)

FARMERS_ADJ = bool(config["adjust_for_farmers"])

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = bool(config["output_intermediate_outputs"])

LOGGER = lu_logging.configure_logger(OUTPUT_DIR / OutputLevel.SUPPORTING, log_name='employment')

# copy config file for traceability
shutil.copy(
    src=args.config_file,
    dst=OUTPUT_DIR / OutputLevel.SUPPORTING / args.config_file.name
)

# --- Step 0 --- #
# read in the data from the config file
block = 'base_data'
LOGGER.info("Importing data from config file")

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

# enusre lad is in known zone system
lad_raw_4_digit_sic = lad_raw_4_digit_sic.translate_zoning(
    new_zoning=constants.LAD_EWS_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.SPATIAL,
    check_totals=False,
)

# --- Useful functions that probably should be DVector methods --- #
def drop_seg_values(dvec: DVector, drop_values: list[int]) -> DVector:
    """Drop rows with provided seg values keep other rows (requires dvector to be single index)
    Args:
        dvec (DVector): DVector function will be applied to
        drop_values (list[int]): values to drop
    Returns:
        DVector: Dvector with values removed
    """

    return DVector(
        segmentation=dvec.segmentation,
        import_data=dvec.data.drop(drop_values),
        zoning_system=dvec.zoning_system,
        cut_read=True,
    )

def keep_seg_values(dvec: DVector, keep_values: list[int]) -> DVector:
    """Keep rows with provided seg values drop other rows (requires dvector to be single index)
    Args:
        dvec (DVector): DVector function will be applied to
        keep_values (list[int]): segmentation values to keep
    Returns:
        DVector: Dvector with values removed
    """

    return DVector(
        segmentation=dvec.segmentation,
        import_data=dvec.data.loc[keep_values],
        zoning_system=dvec.zoning_system,
        cut_read=True,
    )

employment_redistri_lsoa = data_processing.read_dvector_from_config(
    config=config, 
    data_block=block, 
    key='employment_lsoa_distribution_factors'
)

# --- Step 0 --- #
LOGGER.info('--- Step 0 ---')
LOGGER.info(
    'Balance the input datasets to each have the same totals at LAD level as the 4 Digit SIC 2022 Raw data'
)

def constrain_to_lad_totals_w_farmers_adj(
        lad_dv:DVector, 
        msoa_dv:DVector, 
        lsoa_dv:DVector
    ) -> tuple[DVector, DVector]:
    LOGGER.info(
        'parameters specify that we want to apply farmers adjustment (taking farmers numbers from lad and distributing to msoa/lsoa)'
    )
    LOGGER.info(
        'information for farmers (sic 4 digit = 1) taken from LAD input data'
    )

    # check to see if farmers are mentioned in lad data, if not then halt process
    try:
        lad_not_farmers_sic_4_digit = drop_seg_values(lad_dv, [1])
    except KeyError:
        raise ValueError(f"Category 1 does not appear in LAD input, so farmers correction can not be applied")

    lad_total_no_farmers = lad_not_farmers_sic_4_digit.add_segments(
        [constants.CUSTOM_SEGMENTS["total"]], split_method="split"
    ).aggregate(["total"])

    msoa_2011_1_digit_sic_no_farmers = drop_seg_values(msoa_dv, [1]).translate_zoning(
        new_zoning=constants.LAD_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    msoa_total_at_lad_no_farmers = msoa_2011_1_digit_sic_no_farmers.add_segments(
        [constants.CUSTOM_SEGMENTS["total"]], split_method="split"
    ).aggregate(["total"])

    lsoa_2011_1_digit_sic_no_farmers = drop_seg_values(lsoa_dv, [1]).translate_zoning(
        new_zoning=constants.LAD_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    lsoa_total_at_lad_no_farmers = lsoa_2011_1_digit_sic_no_farmers.add_segments(
        [constants.CUSTOM_SEGMENTS["total"]], split_method="split"
    ).aggregate(["total"])

    # Resulting factors, note these don't include farmers

    # TODO: consider having a output/log of where the increases are outside expectations. Along the lines of
    # sig_increases = adjustment_factors.data[adjustment_factors.data.ge(1.1)]
    # sig_decreases = adjustment_factors.data[adjustment_factors.data.le(0.9)]

    msoa_adj_factors_no_farmers = lad_total_no_farmers / msoa_total_at_lad_no_farmers
    lsoa_adj_factors_no_farmers = lad_total_no_farmers / lsoa_total_at_lad_no_farmers

    rehydrated_adj_factors_for_msoa = (
        msoa_adj_factors_no_farmers.add_segments(["sic_2_digit"])
        .aggregate(["sic_2_digit"])
        .translate_zoning(
            new_zoning=constants.MSOA_2011_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.NO_WEIGHT,
            check_totals=False,
        )
    )

    rehydrated_adj_factors_for_lsoa = (
        lsoa_adj_factors_no_farmers.add_segments(
            ["sic_1_digit"]
        )
        .aggregate(["sic_1_digit"])
        .translate_zoning(
            new_zoning=constants.LSOA_2011_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.NO_WEIGHT,
            check_totals=False,
        )
    )

    # apply proportions and fill in nas with 0 (which will be for the unemployed rows)
    # apply proportions filling in nas with 0
    msoa_2011_2_digit_sic_not_farmers = drop_seg_values(msoa_dv, [1])
    adj_msoa_2011_2_digit_sic = (
        msoa_2011_2_digit_sic_not_farmers * rehydrated_adj_factors_for_msoa
    )
    adj_msoa_2011_2_digit_sic.fillna(0)

    #adj_msoa_2011_2_digit_sic = drop_seg_values(adj_msoa_2011_2_digit_sic, [1])

    # Moving onto the farmers part of the process
    # Find the number of farmers provided at LAD but translate to MSOA
    lad_data_in_msoa = lad_dv.translate_zoning(
        new_zoning=constants.MSOA_2011_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    lad_farmers_in_msoa = keep_seg_values(lad_data_in_msoa, [1])

    lad_farmers_in_msoa_sic_2 = lad_farmers_in_msoa.add_segments(
        [SegmentsSuper("sic_2_digit").get_segment()]
    ).aggregate([SegmentsSuper("sic_2_digit").get_segment().name])

    # Find the number of farmers provided at LAD but translate to LSOA
    lsoa_2011_1_digit_sic_not_farmers = drop_seg_values(lsoa_dv, [1])
    adj_lsoa_2011_1_digit_sic = (
        lsoa_2011_1_digit_sic_not_farmers * rehydrated_adj_factors_for_lsoa
    )
    adj_lsoa_2011_1_digit_sic.fillna(0)

    # adj_lsoa_2011_1_digit_sic = drop_seg_values(adj_lsoa_2011_1_digit_sic, [1])

    # Find the number of farmers provided at LAD but translate to MSOA
    lad_data_in_msoa = lad_dv.translate_zoning(
        new_zoning=constants.MSOA_2011_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    lad_farmers_in_msoa = keep_seg_values(lad_data_in_msoa, [1])

    lad_farmers_in_msoa_sic_2 = lad_farmers_in_msoa.add_segments(
        [SegmentsSuper("sic_2_digit").get_segment()]
    ).aggregate([SegmentsSuper("sic_2_digit").get_segment().name])

    # Find the number of farmers provided at LAD but translate to LSOA
    lad_data_in_lsoa = lad_dv.translate_zoning(
        new_zoning=constants.LSOA_2011_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    lad_farmers_in_lsoa = keep_seg_values(lad_data_in_lsoa, [1])

    lad_farmers_in_lsoa_sic_1 = lad_farmers_in_lsoa.add_segments(
        [SegmentsSuper("sic_1_digit").get_segment()]
    ).aggregate([SegmentsSuper("sic_1_digit").get_segment().name])

    # combining farmers from LAD with balanced non farmers (from MSOA and LSOA) inputs
    # MSOA
    farmers_in_msoa_input = keep_seg_values(msoa_dv, [1])
    farmers_by_msoa = farmers_in_msoa_input + lad_farmers_in_msoa_sic_2
    adj_msoa_2011_2_digit_sic = adj_msoa_2011_2_digit_sic.concat(farmers_by_msoa)

    # LSOA
    farmers_in_lsoa_input = keep_seg_values(lsoa_dv, [1])
    farmers_by_lsoa = farmers_in_lsoa_input + lad_farmers_in_lsoa_sic_1
    adj_lsoa_2011_1_digit_sic = adj_lsoa_2011_1_digit_sic.concat(farmers_by_lsoa)
    return adj_lsoa_2011_1_digit_sic, adj_msoa_2011_2_digit_sic


def constrain_to_lad_totals(
        lad_dv:DVector, 
        msoa_dv:DVector, 
        lsoa_dv:DVector
    ) -> tuple[DVector, DVector]:
    lad_2011_2_digit_sic = (
        msoa_dv.translate_zoning(
            new_zoning=constants.LAD_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False,
        )
    )

    lad_2011_1_digit_sic = (
        lsoa_dv.translate_zoning(
            new_zoning=constants.LAD_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False,
        )
    )

    lad_total = lad_dv.add_segments(
        [constants.CUSTOM_SEGMENTS['total']], split_method='split'
    ).aggregate(['total'])

    msoa_total_at_lad = lad_2011_2_digit_sic.add_segments(
        [constants.CUSTOM_SEGMENTS['total']], split_method='split'
    ).aggregate(['total'])

    lsoa_total_at_lad = lad_2011_1_digit_sic.add_segments(
        [constants.CUSTOM_SEGMENTS['total']], split_method='split'
    ).aggregate(['total'])

    msoa_adj_factors = lad_total / msoa_total_at_lad

    lsoa_adj_factors = lad_total / lsoa_total_at_lad

    # TODO: consider having a output/log of where the increases are outside expectations. Along the lines of
    # sig_increases = adjustment_factors.data[adjustment_factors.data.ge(1.1)]
    # sig_decreases = adjustment_factors.data[adjustment_factors.data.le(0.9)]

    rehydrated_adj_factors_for_msoa = (
        msoa_adj_factors.add_segments([SegmentsSuper('sic_2_digit').get_segment()])
        .aggregate([SegmentsSuper('sic_2_digit').get_segment().name])
        .translate_zoning(
            new_zoning=constants.MSOA_2011_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER, 
            weighting=TranslationWeighting.NO_WEIGHT,
            check_totals=False,
        )
    )

    rehydrated_adj_factors_for_msoa = drop_seg_values(
        rehydrated_adj_factors_for_msoa, 
        drop_values=[-1]
    )

    rehydrated_adj_factors_for_lsoa = (
        lsoa_adj_factors.add_segments([SegmentsSuper('sic_1_digit').get_segment()])
        .aggregate([SegmentsSuper('sic_1_digit').get_segment().name])
        .translate_zoning(
            new_zoning=constants.LSOA_2011_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.NO_WEIGHT,
            check_totals=False,
        )
    )

    rehydrated_adj_factors_for_lsoa = drop_seg_values(
        rehydrated_adj_factors_for_lsoa, 
        drop_values=[-1]
    )

    msoa_dv = msoa_dv.translate_zoning(
            new_zoning=rehydrated_adj_factors_for_msoa.zoning_system,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.NO_WEIGHT,
            check_totals=False,
    )
    
    lsoa_dv = lsoa_dv.translate_zoning(
            new_zoning=rehydrated_adj_factors_for_lsoa.zoning_system,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.NO_WEIGHT,
            check_totals=False,
    )

    # apply proportions and fill in nas with 0 (which will be for the unemployed rows)
    adj_msoa_2011_2_digit_sic = rehydrated_adj_factors_for_msoa * msoa_dv
    adj_msoa_2011_2_digit_sic.fillna(0)

    adj_lsoa_2011_1_digit_sic = rehydrated_adj_factors_for_lsoa * lsoa_dv
    adj_lsoa_2011_1_digit_sic.fillna(0)
    return adj_lsoa_2011_1_digit_sic, adj_msoa_2011_2_digit_sic

if FARMERS_ADJ:
    constraint_func = constrain_to_lad_totals_w_farmers_adj
else:
    constraint_func = constrain_to_lad_totals

# constraining to lad totals
adj_lsoa_2011_1_digit_sic, adj_msoa_2011_2_digit_sic = constraint_func(
    lad_dv=lad_raw_4_digit_sic, 
    msoa_dv=msoa_raw_2_digit_sic, 
    lsoa_dv=lsoa_raw_1_digit_sic
)

# --- Step 1 --- #
LOGGER.info('--- Step 1 ---')
LOGGER.info('Exporting district-based 4 Digit SIC 2022 data (Output E1)')
# save output to hdf and csvs for checking
data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference='Output E1',
        dvector=lad_raw_4_digit_sic,
        dvector_dimension='jobs',
        output_level=OutputLevel.INTERMEDIATE
)

# --- Step 2 --- #
LOGGER.info('--- Step 2 ---')
LOGGER.info('Convert 2 Digit SIC 2022 data held in MSOA 2011 zoning to 2021 MSOA (Output E2)')
# LAD is already at LAD 2021 zoning so doesn't need translating
msoa_2021_2_digit_sic = adj_msoa_2011_2_digit_sic.translate_zoning(
        new_zoning=constants.MSOA_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=True
)

# save output to hdf and csvs for checking
data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference='Output E2',
        dvector=msoa_2021_2_digit_sic,
        dvector_dimension='jobs',
        output_level=OutputLevel.INTERMEDIATE
)


# --- Step 3 --- #
LOGGER.info('--- Step 3 ---')
LOGGER.info('Convert 1 Digit SIC 2022 data held in LSOA 2011 zoning to 2021 LSOA (Output E3)')
lsoa_2021_1_digit_sic = adj_lsoa_2011_1_digit_sic.translate_zoning(
        new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=True
)

# save output to hdf and csvs for checking
data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference='Output E3',
        dvector=lsoa_2021_1_digit_sic,
        dvector_dimension='jobs',
        output_level=OutputLevel.INTERMEDIATE
)

# --- Step 4 --- #
LOGGER.info('--- Step 4 ---')
LOGGER.info(f'Converting SIC SOC jobs from Region to LSOA 2021 level (Output E4)')
ons_sic_soc_jobs_lsoa = ons_sic_soc_jobs_lu.translate_zoning(
    new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.NO_WEIGHT,
    check_totals=False
)

# Note a warning is generated here about combinations with SOC as 4. We can ignore it.
LOGGER.info(f'Applying SOC group proportions to 1-digit SIC jobs')
jobs_by_lsoa_with_soc_group = data_processing.apply_proportions(
    source_dvector=ons_sic_soc_jobs_lsoa,
    apply_to=lsoa_2021_1_digit_sic
)

LOGGER.info('Converting proportions of SIC 2 digit by SIC 1 digit by SOC groups jobs to LSOA 2021')

lsoa_1_digit_2_digit_sic = msoa_2021_2_digit_sic.add_segments(
    [SegmentsSuper("sic_1_digit").get_segment()])

msoa_1_digit_2_digit_sic = lsoa_1_digit_2_digit_sic.translate_zoning(
    new_zoning=jobs_by_lsoa_with_soc_group.zoning_system,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.SPATIAL,
    check_totals=False,
)

# Note a warning is generated here about combinations with SOC as 4. We can ignore it.
LOGGER.info(f'Applying SOC group proportions to 2-digit SIC jobs')
jobs_by_sic_soc_lsoa_no_soc_4 = data_processing.apply_proportions(
    source_dvector=msoa_1_digit_2_digit_sic,
    apply_to=jobs_by_lsoa_with_soc_group
)

# now need to expand the data to include soc 4 (unemployed) based on factors
totals = jobs_by_sic_soc_lsoa_no_soc_4.add_segments(
    [constants.CUSTOM_SEGMENTS['total']]).aggregate(['total'])

totals_with_segs = totals.add_segments([
    SegmentsSuper('sic_1_digit').get_segment(),
        SegmentsSuper('sic_2_digit').get_segment(),
        SegmentsSuper('soc').get_segment()]
).aggregate([
    'sic_1_digit', 'sic_2_digit', 'soc'])

soc_1_3_totals = totals_with_segs.filter_segment_value('sic_1_digit', -1)

soc_1_3_totals = soc_1_3_totals.add_segments([SegmentsSuper('sic_1_digit').get_segment()])

# TODO: switch to using translate zoning directly, 
# however currently doesn't work with one row df, needs fix in caf.toolkit
# soc_4_factors_lsoa = soc_4_factors.translate_zoning(
#         new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
#         cache_path=constants.CACHE_FOLDER,
#         weighting=TranslationWeighting.NO_WEIGHT,
#         check_totals=True,
# )

#### workaround start ################################################################
soc_4_factors_with_soc = soc_4_factors.add_segments(["soc"])

soc_4_factors_with_soc_lsoa = soc_4_factors_with_soc.translate_zoning(
        new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False,
)

soc_4_factors_lsoa = soc_4_factors_with_soc_lsoa.aggregate(["total"])

soc_4_factors_lsoa.data = soc_4_factors_lsoa.data / len(soc_4_factors_with_soc_lsoa.data)

#### workaround end ##################################################################

# sort out the segmentations to match the full dataset
soc_4_factors_lsoa = (
    soc_4_factors_lsoa
    .add_segments(["sic_2_digit", "soc", "sic_1_digit"])
    .aggregate(["sic_2_digit", "soc", "sic_1_digit"])
    .filter_segment_value("sic_2_digit", -1)
    )

# apply factors to soc 1-3 totals to give the number of soc_4 people by lsoa
soc_4_row = soc_1_3_totals * soc_4_factors_lsoa

jobs_by_sic_soc_lsoa = jobs_by_sic_soc_lsoa_no_soc_4.concat(soc_4_row)

# save output to hdf and csvs for checking
data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference='Output E4',
        dvector=jobs_by_sic_soc_lsoa,
        dvector_dimension='jobs',
        output_level=OutputLevel.FINAL
)

LOGGER.info(f'Uplifting Output E4 to Workforce jobs (WFJ) levels by region (Output E4_2)')

output_e4_by_rgn = jobs_by_sic_soc_lsoa_no_soc_4.translate_zoning(
    new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.SPATIAL,
    check_totals=False
)

e4_total_by_rgn = output_e4_by_rgn.add_segments(
    [constants.CUSTOM_SEGMENTS['total']], split_method='split'
).aggregate(['total'])

factors = wfj / e4_total_by_rgn

rehydrated_adj_factors_for_e4_2 = (
    factors.add_segments([
        SegmentsSuper('sic_1_digit').get_segment(),
        SegmentsSuper('sic_2_digit').get_segment(),
        SegmentsSuper('soc').get_segment()
    ])
    .aggregate([
        SegmentsSuper('sic_1_digit').get_segment().name,
        SegmentsSuper('sic_2_digit').get_segment().name,
        SegmentsSuper('soc').get_segment().name
        ])
    .translate_zoning(
        new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )
)

# Apply WFJ adjustment factors to *all* population - i.e. we assume Unemployed get the same uplift as employed.
output_e4_2 = jobs_by_sic_soc_lsoa * rehydrated_adj_factors_for_e4_2

data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference='Output E4_2',
        dvector=output_e4_2,
        dvector_dimension='jobs',
        output_level=OutputLevel.FINAL
)

LOGGER.info("Adjusting Output E4 to adjust distributions within LADs (Output E4.3)")
LOGGER.info("This is using a combination of voa floorspace and pupil distribution")
LOGGER.info("Which varies by sic and lsoa, and is only applied to previously defined regions")

# Need to use the E4 output before the soc4 row is added. As jobs will be moved between lsoa this row must be recalculated.

lad_totals = jobs_by_sic_soc_lsoa_no_soc_4.translate_zoning(
    new_zoning=constants.LAD_EWS_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.SPATIAL,
    check_totals=False,
)

lad_at_lsoa = lad_totals.translate_zoning(
    new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.NO_WEIGHT,
    check_totals=False,
)
# multiply lad totals by splits for that sic/lsoa combination. Note resulting DVector will have lots of nas
# the nas will correspond to where we do not wish to update the distribution for that sic/lsoa.

# TODO: see if we can avoid the na rows being dropped and if so could switch to the more straightforward
# adj_jobs_lsoa_no_soc_4 = lad_at_lsoa * employment_redistri_lsoa

adj_jobs_lsoa_no_soc_4 = lad_at_lsoa.copy()
adj_jobs_lsoa_no_soc_4.data = lad_at_lsoa.data * employment_redistri_lsoa.data

# infill the nas back with the original values
adj_jobs_lsoa_no_soc_4.data = adj_jobs_lsoa_no_soc_4.data.fillna(
    jobs_by_sic_soc_lsoa_no_soc_4.data
)
# fix back to floats introduced as a result of nas being introduced
adj_jobs_lsoa_no_soc_4.data = adj_jobs_lsoa_no_soc_4.data.astype(float)

# now need to expand the data to include soc 4 (unemployed) based on factors
totals = adj_jobs_lsoa_no_soc_4.add_segments(
    [constants.CUSTOM_SEGMENTS["total"]]
).aggregate(["total"])

totals_with_segs = totals.add_segments(
    [
        SegmentsSuper("sic_1_digit").get_segment(),
        SegmentsSuper("sic_2_digit").get_segment(),
        SegmentsSuper("soc").get_segment(),
    ]
).aggregate(["sic_1_digit", "sic_2_digit", "soc"])

soc_1_3_totals = totals_with_segs.filter_segment_value("sic_1_digit", -1)

soc_1_3_totals = soc_1_3_totals.add_segments(
    [SegmentsSuper("sic_1_digit").get_segment()]
)

# soc 4 factors remain the same as before
soc_4_row = soc_1_3_totals * soc_4_factors_lsoa

adj_jobs_sic_soc_lsoa = adj_jobs_lsoa_no_soc_4.concat(soc_4_row)

data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference="Output E4_3",
    dvector=adj_jobs_sic_soc_lsoa,
    dvector_dimension="jobs",
    output_level=OutputLevel.FINAL
)

LOGGER.info('--- Step 5 ---')
LOGGER.info(f'Combining Output E1 (+soc 4) and E4 to give Jobs by LSOA SIC 4 digit and SOC group (1-4) (Output E5)')
# Output E5

# First need to create a Dummy DVector of 1's to concat to E1
columns = list(lad_raw_4_digit_sic.data.columns)

# infill with 1's as we will be taking all the proportion for these rows from e4
e1_soc_4_row = pd.DataFrame(1.0, index=range(1), columns=columns)
e1_soc_4_row["sic_4_digit"] = -1
e1_soc_4_row = e1_soc_4_row.set_index("sic_4_digit")

seg_new = cc.Segmentation(
    cc.SegmentationInput(
        enum_segments=["sic_4_digit"],
        naming_order=["sic_4_digit"],
        subsets={"sic_4_digit": [-1]},
    )
)
soc_4_dummy = DVector(
    segmentation=seg_new,
    import_data=e1_soc_4_row,
    zoning_system=lad_raw_4_digit_sic.zoning_system,
)
# Add on dummy row for soc 4
adjusted_e1 = lad_raw_4_digit_sic.concat(soc_4_dummy)

# Now having got the adjusted e1 we can prepare it for merging with e4
# e1_with_sic_2_lad = adjusted_e1.translate_segment(
#     from_seg=SegmentsSuper('sic_4_digit').get_segment(),
#     to_seg=SegmentsSuper('sic_2_digit').get_segment(),
#     drop_from=False
# )
e1_with_sic_2_lad = adjusted_e1.translate_segment(
    from_seg='sic_4_digit',
    to_seg='sic_2_digit',
    drop_from=False
)

e1_with_sic_2_lsoa = e1_with_sic_2_lad.translate_zoning(
    new_zoning=constants.LSOA_EWS_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.SPATIAL,
    check_totals=False
)

# apply proportions, including getting soc 4 values to match e4
jobs_by_sic_2_4_soc_lsoa = data_processing.apply_proportions(
    source_dvector=e1_with_sic_2_lsoa, 
    apply_to=jobs_by_sic_soc_lsoa
)

data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference='Output E5',
        dvector=jobs_by_sic_2_4_soc_lsoa,
        dvector_dimension='jobs',
        output_level=OutputLevel.FINAL
)

LOGGER.info(
    f"Combining Output E1 (+soc 4) and E4.3 to give Jobs by LSOA SIC 4 digit and SOC group (1-4) (Output E6)"
)
LOGGER.info(
    f"This will take account of the redistribution of jobs using voa floorspace and pupils"
)

adj_jobs_by_sic_2_4_soc_lsoa = data_processing.apply_proportions(
    source_dvector=e1_with_sic_2_lsoa, apply_to=adj_jobs_sic_soc_lsoa
)

data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference="Output E6",
    dvector=adj_jobs_by_sic_2_4_soc_lsoa,
    dvector_dimension="jobs",
    output_level=OutputLevel.FINAL
)

