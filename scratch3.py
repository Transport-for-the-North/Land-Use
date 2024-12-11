from pathlib import Path

import pandas as pd
from caf.base import DVector
import yaml
import numpy as np
from caf.base.zoning import TranslationWeighting

from land_use import data_processing, constants
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel

# config file
config_file = r'scenario_configurations\iteration_5\base_population_config_occupancy_test.yml'
# load configuration file
with open(config_file, 'r') as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

OUTPUT_DIR = Path(config['output_directory'])
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOGGER = lu_logging.configure_logger(
    output_dir=OUTPUT_DIR / OutputLevel.SUPPORTING,
    log_name='population'
)

# read in 2021 base year population and households
census_input_path = Path(r'F:\Working\Land-Use\241206_hh-test\01_Intermediate Files')
census_pop = DVector.load(census_input_path / 'Output P10_NW.hdf')
census_hh = DVector.load(census_input_path / 'Output P4.3_NW.hdf')

# read in the population adjustment data from the config file
# population_adjustment = data_processing.read_dvector_from_config(
#     config=config,
#     data_block='population_adjustment_data',
#     key='rebase_data',
#     geography_subset='NW'
# )
# # bring in segmentations to maintain from the 2021 build datasets
# # try looking for `rebase_segments_to_maintain` key and log if not provided
# rebase_segments_to_maintain = config['population_adjustment_data'].get(
#     'rebase_segments_to_maintain', []
# )
#
# # loop through the supplied segmentations, aggregating the 2021 population
# # data to each of the segmentations provided and deriving a monovariate
# # target for the IPF and adding it to the start if the rebase targets
# added_targets = []
# for segmentation in rebase_segments_to_maintain:
#     if isinstance(segmentation, list):
#         target = census_pop.aggregate(segs=segmentation)
#     else:
#         target = census_pop.aggregate(segs=[segmentation])
#     added_targets.append(target)
#
# population_adjustment_targets = added_targets + population_adjustment
#
# # rebase population
# rebased_pop, summary, differences = data_processing.apply_ipf(
#     seed_data=census_pop,
#     target_dvectors=population_adjustment_targets,
#     cache_folder=constants.CACHE_FOLDER,
#     # todo change
#     target_dvector=population_adjustment[0],
# )
# rebased_pop.save(OUTPUT_DIR / '2023_pop.hdf')
rebased_pop = DVector.load(OUTPUT_DIR / '2023_pop.hdf')

# read in 2021 to 2023 household growth
growth = data_processing.read_dvector_data(
    file_path=Path(r'RM002 accom type by household size\preprocessing\Household_growth_checks.hdf'),
    geographical_level=constants.LAD_NAME,
    input_segments=['total'],
    geography_subset='NW',
    input_root_directory=config['input_root_directory']
)

# get total population by LAD for the rebased pop
rebased_pop_total = rebased_pop.add_segments(["total"]).aggregate(["total"]).translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-NW'),
    cache_path=constants.CACHE_FOLDER
)

# get total population by LAD for the census pop
census_pop_total = census_pop.add_segments(["total"]).aggregate(["total"]).translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-NW'),
    cache_path=constants.CACHE_FOLDER
)

# calculate population growth by LAD
pop_growth = rebased_pop_total / census_pop_total

# calculate household growth control factor at district level to make sure
# total growth does not go beyond what the 2021-to-2023 forecasts predict
growth_control_factor = growth / pop_growth

# convert back to LSOA
growth_control_factor_lsoa = growth_control_factor.translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-NW'),
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.NO_WEIGHT,
    check_totals=False
)

# calculate household growth by household type by LSOA by grouping population
# and applying the control factor
rebase_households = ((rebased_pop.aggregate(["accom_h"]) / census_pop.aggregate(["accom_h"])).add_segments(
    ["total"]
) * growth_control_factor_lsoa).aggregate(['accom_h']) * census_hh

# add total segmentation to 2021 households and aggregate
census_hh_total = census_hh.add_segments(
    ["total"]
).aggregate(['total'])

# derive household total targets based on growth applied to 2021 households
# at district level
rebase_household_target = census_hh_total.translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-NW'),
    cache_path=constants.CACHE_FOLDER
) * growth

# derive household constraints from population data (required occupancies)
household_occupancy_1_target = rebased_pop.aggregate(['adults', 'children']).filter_segment_value(
    segment_name='adults', segment_values=[1]
).filter_segment_value(
    segment_name='children', segment_values=[1]
)
household_occupancy_2_target = rebased_pop.aggregate(['adults', 'children']).filter_segment_value(
    segment_name='adults', segment_values=[2]
).filter_segment_value(
    segment_name='children', segment_values=[1]
) / 2

household_adult_1_target = rebased_pop.aggregate(['adults', 'age_9']).filter_segment_value(
    segment_name='age_9', segment_values=[4, 5, 6, 7, 8, 9]
).filter_segment_value(
    segment_name='adults', segment_values=[1]
).aggregate(['adults'])
household_adult_2_target = (rebased_pop.aggregate(['adults', 'age_9']).filter_segment_value(
    segment_name='age_9', segment_values=[4, 5, 6, 7, 8, 9]
).filter_segment_value(
    segment_name='adults', segment_values=[2]
) / 2).aggregate(['adults'])

household_adjustment_targets = [
    rebase_household_target, household_occupancy_1_target, household_occupancy_2_target,
    household_adult_1_target, household_adult_2_target
]
rebased_households, summary, differences = data_processing.apply_ipf(
    seed_data=rebase_households,
    target_dvectors=household_adjustment_targets,
    cache_folder=constants.CACHE_FOLDER,
    # todo change
    # target_dvector=rebase_households.aggregate(['accom_h']),
)
rebased_households.save(OUTPUT_DIR / '2023_hh.hdf')

# output adhoc stuff for analysis
rebased = rebased_households.data.reset_index().melt(
    id_vars=list(rebased_households.data.index.names),
    value_vars=list(rebased_households.data.columns),
    var_name='LSOA', value_name='households'
)
rebased.to_csv(
    r"F:\Working\Land-Use\241210_occupancy checks\2023_hh.csv", index=False
)

census = census_hh.data.reset_index().melt(
    id_vars=list(census_hh.data.index.names),
    value_vars=list(census_hh.data.columns),
    var_name='LSOA',
    value_name='households'
)
census.to_csv(
    r"F:\Working\Land-Use\241210_occupancy checks\2021_hh.csv", index=False
)

# calculate and output occupancies
resulting_occupancies = rebased_pop.aggregate(['adults', 'children']) / rebased_households.aggregate(['adults', 'children'])
resulting_occupancies.save(OUTPUT_DIR / '2023_occupancies.hdf')
occupancies = resulting_occupancies.data.reset_index().melt(
    id_vars=list(resulting_occupancies.data.index.names),
    value_vars=list(resulting_occupancies.data.columns),
    var_name='LSOA',
    value_name='occupancy'
)
occupancies.to_csv(
    r"F:\Working\Land-Use\241210_occupancy checks\2023_occupancies.csv", index=False
)

resulting_occupancies = rebased_pop.filter_segment_value(
    segment_name='age_9', segment_values=[4, 5, 6, 7, 8, 9]
).aggregate(['adults']) / rebased_households.aggregate(['adults'])
resulting_occupancies.save(OUTPUT_DIR / '2023_adult_occupancies.hdf')
occupancies = resulting_occupancies.data.reset_index().melt(
    id_vars=list(resulting_occupancies.data.index.names),
    value_vars=list(resulting_occupancies.data.columns),
    var_name='LSOA',
    value_name='occupancy'
)
occupancies.to_csv(
    r"F:\Working\Land-Use\241210_occupancy checks\2023_adult_occupancies.csv", index=False
)

# calculate resulting household growth by district
resulting_household_growth = rebased_households.aggregate(['total']).translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-NW'),
    cache_path=constants.CACHE_FOLDER
) / census_hh.add_segments(
    ["total"]
).aggregate(['total']).translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-NW'),
    cache_path=constants.CACHE_FOLDER
)
resulting_household_growth = resulting_household_growth.data.reset_index().melt(
    id_vars=list(resulting_household_growth.data.index.names),
    value_vars=list(resulting_household_growth.data.columns),
    var_name='LAD',
    value_name='growth'
)
resulting_household_growth.to_csv(
    r"F:\Working\Land-Use\241210_occupancy checks\2023_output_growth.csv", index=False
)

# calculate input household growth by district
input_household_growth = growth.data.reset_index().melt(
    id_vars=list(growth.data.index.names),
    value_vars=list(growth.data.columns),
    var_name='LAD',
    value_name='growth'
)
input_household_growth.to_csv(
    r"F:\Working\Land-Use\241210_occupancy checks\2023_input_growth.csv", index=False
)

