from argparse import ArgumentParser
from functools import reduce
from pathlib import Path
from typing import Tuple
import shutil

import yaml
from caf.base import DVector
from caf.base.segments import SegmentsSuper
from caf.base.zoning import TranslationWeighting
import numpy as np

from land_use import constants, data_processing
from land_use import logging as lu_logging
from land_use.data_processing import OutputLevel, BaseYearPopulationData


def process_base(
        configuration: dict,
        geography_subset: str
) -> BaseYearPopulationData:
    # --- Step 0 --- #
    # read in the base data from the config file
    block = 'base_data'
    LOGGER.info(f'Importing base data from config file ({block} block)')
    occupied_households = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='occupied_households',
        geography_subset=geography_subset
    )
    unoccupied_households = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='unoccupied_households',
        geography_subset=geography_subset
    )
    ons_table_1 = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ons_table_1',
        geography_subset=geography_subset
    )
    addressbase_dwellings = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='addressbase_dwellings',
        geography_subset=geography_subset
    )
    ons_table_2 = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ons_table_2',
        geography_subset=geography_subset
    )
    ons_table_4 = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ons_table_4',
        geography_subset=geography_subset
    )
    hh_age_gender_2021 = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='hh_age_gender_2021',
        geography_subset=geography_subset
    )
    ons_table_3 = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ons_table_3',
        geography_subset=geography_subset
    )
    ce_uplift_factor = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ce_uplift_factor',
        geography_subset=geography_subset
    )
    ce_pop_by_type = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ce_pop_by_type',
        geography_subset=geography_subset
    )
    ce_pop_by_age_gender_soc = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ce_pop_by_age_gender_soc',
        geography_subset=geography_subset
    )
    ce_pop_by_age_gender_econ = data_processing.read_dvector_from_config(
        config=configuration,
        data_block=block,
        key='ce_pop_by_age_gender_econ',
        geography_subset=geography_subset
    )

    # read in the household validation data from the config file
    LOGGER.info(f'Importing household validation data from config file ({block} block)')
    household_adjustment = data_processing.read_dvector_from_config(
        config=configuration,
        data_block='household_adjustment_data',
        key='validation_data',
        geography_subset=geography_subset
    )

    # read in the population adjustment data from the config file
    LOGGER.info(f'Importing population adjustment data from config file ({block} block)')
    population_adjustment = data_processing.read_dvector_from_config(
        config=configuration,
        data_block='population_adjustment_data',
        key='validation_data',
        geography_subset=geography_subset
    )

    # --- Step 1 --- #
    LOGGER.info('--- Step 1 ---')
    LOGGER.info(f'Calculating average occupancy by dwelling type')
    # Create a total dvec of total number of households based on occupied_properties + unoccupied_properties
    all_properties = unoccupied_households + occupied_households

    # Calculate adjustment factors by zone to get proportion of households occupied by dwelling type by zone
    non_empty_proportion = occupied_households / all_properties

    # infill missing adjustment factors with average value of other properties
    # in the LSOA. Note this is where the total households in and LSOA of a given
    # type is 0
    # TODO do we want to do anything about 1/0 proportions??
    non_empty_proportion.data = non_empty_proportion.data.fillna(
        non_empty_proportion.data.mean(axis=0), axis=0
    )

    # average occupancy for occupied dwellings
    # TODO this average occupancy is now based on census households, not addressbase, is this what we want? Or do we want to use the adjusted addressbase?
    average_occupancy = (ons_table_1 / occupied_households)

    # replace infinities with nans for infilling
    # this is where the occupied_households value is zero for a dwelling type and LSOA,
    # but the ons_table_1 has non-zero population. E.g. LSOA E01007423 in gor = 'YH'
    # caravans and mobile homes, the occupied households = 0 but ons_table_1 population = 4
    average_occupancy._data = average_occupancy._data.replace(np.inf, np.nan)

    # infill missing occupancies with average value of other properties in the LSOA
    # i.e. based on column
    average_occupancy._data = average_occupancy._data.fillna(
        average_occupancy._data.mean(axis=0), axis=0
    )

    # calculate unoccupied households as a function of occupied to get 2023
    # unoccupied households later on
    unoccupied_factor = unoccupied_households / occupied_households
    unoccupied_factor._data = unoccupied_factor._data.replace(np.inf, np.nan)
    unoccupied_factor._data = unoccupied_factor._data.fillna(
        unoccupied_factor._data.mean(axis=0), axis=0
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P1.1_{geography_subset}',
        dvector=occupied_households,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P1.2_{geography_subset}',
        dvector=unoccupied_households,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P1.3_{geography_subset}',
        dvector=average_occupancy,
        dvector_dimension='occupancy',
        output_level=OutputLevel.INTERMEDIATE
    )
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P1.4_{geography_subset}',
        dvector=non_empty_proportion,
        dvector_dimension='factor',
        output_level=OutputLevel.INTERMEDIATE
    )
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P1.5_{geography_subset}',
        dvector=unoccupied_factor,
        dvector_dimension='factor',
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        occupied_households, unoccupied_households
    )

    # --- Step 2 --- #
    LOGGER.info('--- Step 2 ---')
    LOGGER.info(f'Adjusting addressbase buildings to reflect unoccupied dwellings')

    # apply factors of proportion of total households that are occupied by LSOA
    adjusted_addressbase_dwellings = addressbase_dwellings * non_empty_proportion

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P2_{geography_subset}',
        dvector=adjusted_addressbase_dwellings,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        addressbase_dwellings
    )

    # --- Step 3 --- #
    LOGGER.info('--- Step 3 ---')
    LOGGER.info(f'Applying NS-SeC proportions to Adjusted AddressBase dwellings')
    # apply proportional factors based on hh ns_sec to the addressbase dwellings
    hh_by_nssec = data_processing.apply_proportions(ons_table_4, adjusted_addressbase_dwellings)

    # check against original addressbase data
    # check = hh_by_nssec.aggregate(segs=['accom_h'])

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P3_{geography_subset}',
        dvector=hh_by_nssec,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        ons_table_4
    )

    # --- Step 4 --- #
    LOGGER.info('--- Step 4 ---')

    LOGGER.info('Converting ONS Table 2 to LSOA level (only to be used in proportions, totals will be wrong)')
    # expand these factors to LSOA level
    ons_table_2_lsoa = ons_table_2.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    # check proportions sum to one
    # tmp = proportion_hhs_by_h_hc_ha_car_lsoa.aggregate(segs=['accom_h'])

    LOGGER.info(f'Applying children, adult, and car availability proportions to households')
    # apply proportional factors based on hh by adults / children / car availability to the hh by nssec
    hh_by_nssec_hc_ha_car = data_processing.apply_proportions(ons_table_2_lsoa, hh_by_nssec)

    hh_by_nssec_hc_ha_car = hh_by_nssec_hc_ha_car.add_segments([SegmentsSuper.get_segment(SegmentsSuper.ADULT_NSSEC)])

    # check against original addressbase data
    # check = hh_by_nssec_hc_ha_car.aggregate(segs=['accom_h'])

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P4.1_{geography_subset}',
        dvector=hh_by_nssec_hc_ha_car,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )

    # prepare ons_table_2 for ipf targets (drop accom_h segmentation)
    ons_table_2_target = ons_table_2.aggregate(
        segs=[seg for seg in ons_table_2.data.index.names if seg != 'accom_h']
    )

    # applying IPF
    LOGGER.info('Applying IPF for internal validation household targets')
    internal_rebalanced_hh, summary, differences = data_processing.apply_ipf(
        seed_data=hh_by_nssec_hc_ha_car,
        target_dvectors=[ons_table_2_target],
        cache_folder=constants.CACHE_FOLDER
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P4.2_{geography_subset}',
        dvector=internal_rebalanced_hh,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )
    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f'Output P4.2_{geography_subset}_VALIDATION.csv',
        float_format='%.5f', index=False
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE,
        file=f'Output P4.2_{geography_subset}_VALIDATION.xlsx',
        dfs=differences
    )

    # applying IPF
    LOGGER.info('Applying IPF for independent household targets')
    rebalanced_hh, summary, differences = data_processing.apply_ipf(
        seed_data=internal_rebalanced_hh,
        target_dvectors=list(household_adjustment),
        cache_folder=constants.CACHE_FOLDER
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P4.3_{geography_subset}',
        dvector=rebalanced_hh,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )
    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f'Output P4.3_{geography_subset}_VALIDATION.csv',
        float_format='%.5f', index=False
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE,
        file=f'Output P4.3_{geography_subset}_VALIDATION.xlsx',
        dfs=differences
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        ons_table_2_target, ons_table_2, hh_by_nssec_hc_ha_car,
        internal_rebalanced_hh
    )

    # --- Step 5 --- #
    LOGGER.info('--- Step 5 ---')
    LOGGER.info(f'Applying average occupancy to households')
    # Apply average occupancy by dwelling type to the households by NS-SeC,
    # car availability, number of adults and number of children
    # TODO Do we want to do this in a "smarter" way? The occupancy of 1 adult households (for example) should not be more than 1
    # TODO and households with 2+ children should be more than 3 - is this a place for IPF?
    pop_by_nssec_hc_ha_car = rebalanced_hh * average_occupancy

    # calculate expected population based in the addressbase "occupied" dwellings
    addressbase_population = adjusted_addressbase_dwellings * average_occupancy

    # TODO: Review this. This step will correct the zone totals to match what's in our uplifted AddressBase. Is this going to give the correct number?
    # Rebalance the zone totals
    data_processing.rebalance_zone_totals(
        input_dvector=pop_by_nssec_hc_ha_car,
        desired_totals=addressbase_population
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P5_{geography_subset}',
        dvector=pop_by_nssec_hc_ha_car,
        dvector_dimension='population',
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        addressbase_population, adjusted_addressbase_dwellings
    )

    # --- Step 6 --- #
    LOGGER.info('--- Step 6 ---')
    LOGGER.info(f'Converting household age and gender figures to LSOA level '
                f'(only to be used in proportions, totals will be wrong)')
    # convert to LSOA
    hh_age_gender_2021_lsoa = hh_age_gender_2021.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    LOGGER.info(f'Applying age and gender splits by dwelling type')
    # apply the splits at LSOA level to main population table
    pop_by_nssec_hc_ha_car_gender_age = data_processing.apply_proportions(
        hh_age_gender_2021_lsoa, pop_by_nssec_hc_ha_car
    )

    # compare each step
    data_processing.compare_dvectors(
        dvec1=pop_by_nssec_hc_ha_car,
        dvec2=pop_by_nssec_hc_ha_car_gender_age
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P6_{geography_subset}',
        dvector=pop_by_nssec_hc_ha_car_gender_age,
        dvector_dimension='population',
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        pop_by_nssec_hc_ha_car, hh_age_gender_2021_lsoa
    )

    # --- Step 7 --- #
    LOGGER.info('--- Step 7 ---')
    LOGGER.info('Converting ONS Table 3 to LSOA level '
                '(only to be used in proportions, totals will be wrong)')
    # convert the factors back to LSOA
    ons_table_3_lsoa = ons_table_3.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    # check proportions sum to one
    # TODO some zeros in here that maybe shouldnt be? Need to check
    # tmp = soc_splits_lsoa_age.aggregate(segs=['accom_h', 'age_9', 'ns_sec'])

    # apply the splits at LSOA level to main population table
    LOGGER.info('Applying economic status, employment status, and SOC category '
                'splits to population')
    pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc = data_processing.apply_proportions(
        ons_table_3_lsoa, pop_by_nssec_hc_ha_car_gender_age
    )

    # compare each step
    data_processing.compare_dvectors(
        dvec1=pop_by_nssec_hc_ha_car_gender_age,
        dvec2=pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P7_{geography_subset}',
        dvector=pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc,
        dvector_dimension='population',
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        ons_table_3, ons_table_3_lsoa, pop_by_nssec_hc_ha_car_gender_age
    )

    # --- Step 8 --- #
    LOGGER.info('--- Step 8 ---')
    LOGGER.info(f'Calculating adjustments for communal establishment residents')

    # calculate proportional increase by LSOA due to communal establishments
    # define a matrix of 1s, ce_uplift_factor - 1 doesnt work
    # TODO change dunder method to allow simple numeric operations?
    ones = ce_uplift_factor.copy()
    ones.data.loc[:] = 1
    ce_uplift = ce_uplift_factor - ones

    LOGGER.info(f'Calculating splits of CE type by MSOA')
    # calculate msoa-based splits of CE types
    ce_pop_by_type_total = ce_pop_by_type.add_segments(['total'])
    ce_type_splits = ce_pop_by_type_total / ce_pop_by_type_total.aggregate(segs=['total'])
    # fill in nan values with 0 (this is where there are no CEs in a given MSOA)
    ce_type_splits.data = ce_type_splits.data.fillna(0)

    # translate the MSOA based CE-type splits to LSOA
    ce_type_splits_lsoa = ce_type_splits.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    LOGGER.info(f'Calculating splits of CE population by age, gender, and economic status')
    # calculate gor-based splits of person types
    ce_pop_by_age_gender_econ_total = ce_pop_by_age_gender_econ.add_segments(['total'])
    ce_econ_splits = ce_pop_by_age_gender_econ_total / ce_pop_by_age_gender_econ_total.aggregate(segs=['total'])
    # fill in nan values with 0 (this is where there are no CEs in a given REGION)
    ce_econ_splits.data = ce_econ_splits.data.fillna(0)

    # translate the gor based splits to LSOA
    ce_econ_splits_lsoa = ce_econ_splits.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    LOGGER.info(f'Calculating splits of CE population by age, gender, and SOC')
    # calculate gor-based splits of person types
    ce_pop_by_age_gender_soc_total = ce_pop_by_age_gender_soc.add_segments(['total'])
    ce_soc_splits = ce_pop_by_age_gender_soc_total / ce_pop_by_age_gender_soc_total.aggregate(segs=['total'])
    # fill in nan values with 0 (this is where there are no CEs in a given REGION)
    ce_soc_splits.data = ce_soc_splits.data.fillna(0)

    # translate the gor based splits to LSOA
    ce_soc_splits_lsoa = ce_soc_splits.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    LOGGER.info('Generating CE adjustment dataset at LSOA')
    # split the uplift factor to be by age, gender, soc, econ, and ce type
    LOGGER.info('Applying CE type splits to zonal uplift')
    ce_uplift_by_ce = ce_uplift * ce_type_splits_lsoa
    LOGGER.info('Applying economic status splits to zonal uplift')
    ce_uplift_by_ce_age_gender_econ = ce_uplift_by_ce * ce_econ_splits_lsoa
    LOGGER.info('Applying SOC category splits to zonal uplift')
    ce_uplift_by_ce_age_gender_econ_soc = ce_uplift_by_ce_age_gender_econ * ce_soc_splits_lsoa

    # drop the 'total' segmentation
    ce_uplift_by_ce_age_gender_econ_soc = ce_uplift_by_ce_age_gender_econ_soc.aggregate(
        segs=['ce', 'age_9', 'g', 'economic_status', 'soc']
    )

    # define a matrix of 1s, ce_uplift_by_ce_age_gender_econ_soc + 1 doesnt work
    # TODO change dunder method to allow simple numeric operations?
    ones = ce_uplift_by_ce_age_gender_econ_soc.copy()
    ones.data.loc[:] = 1

    LOGGER.info('Calculating zonal adjustment factors by CE type, age, gender, '
                'economic status, and SOC')
    # calculate adjustment factors by ce type, age, gender, economic status, and SOC
    ce_uplift_factor_by_ce_age_gender_econ_soc = ce_uplift_by_ce_age_gender_econ_soc + ones
    # TODO some level of output is needed here? Confirm with Matteo

    # drop communal establishment type to apply back to main population
    ce_uplift_factor = ce_uplift_by_ce_age_gender_econ_soc.aggregate(
        segs=['age_9', 'g', 'economic_status', 'soc']
    )
    ones = ce_uplift_factor.copy()
    ones.data.loc[:] = 1
    ce_uplift_factor = ce_uplift_factor + ones

    LOGGER.info('Uplifting population to account for CEs')
    # calculate population in CEs by ce type, age, gender, econ status, and soc
    # TODO: This only works *this* way round. Why?
    adjusted_pop = pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc * ce_uplift_factor

    LOGGER.debug('Checks on impact of Communal Establishments uplift')
    ce_change = adjusted_pop - pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc
    data_processing.summary_reporting(
        ce_change,
        dimension='Change derived from Communal Establishments',
    )

    data_processing.compare_dvectors(
        dvec1=pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc,
        dvec2=adjusted_pop
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P8_{geography_subset}',
        dvector=adjusted_pop,
        dvector_dimension='population',
        detailed_logs=True,
        output_level=OutputLevel.INTERMEDIATE
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        ce_pop_by_age_gender_econ_total, ce_pop_by_age_gender_econ,
        ce_econ_splits, ce_econ_splits_lsoa, ce_soc_splits, ce_soc_splits_lsoa,
        ce_uplift_by_ce_age_gender_econ_soc, ones,
        ce_uplift_factor_by_ce_age_gender_econ_soc,
        pop_by_nssec_hc_ha_car_gender_age_econ_emp_soc,
        ce_uplift_factor, ce_change
    )

    # --- Step 9 --- #
    LOGGER.info('--- Step 9 ---')

    # prepare ipf targets (drop accom_h segmentation)
    hh_age_gender_2021_target = hh_age_gender_2021.aggregate(
        segs=[seg for seg in hh_age_gender_2021.data.index.names if seg != 'accom_h']
    )

    # calculate adjustment factor for 2021 population at a total level and apply to the IPF target
    # TODO This is because children ages are small in adjusted_pop and adding as explicit target means these are matched when they probs shouldnt be
    adjustment_factor = adjusted_pop.total / hh_age_gender_2021_target.total
    hh_age_gender_2021_target = hh_age_gender_2021_target * adjustment_factor

    # applying IPF (adjusting totals to match P9 outputs)
    LOGGER.info('Applying IPF for internal validation population targets')
    rebalanced_pop, summary, differences = data_processing.apply_ipf(
        seed_data=adjusted_pop,
        target_dvectors=[hh_age_gender_2021_target],
        cache_folder=constants.CACHE_FOLDER
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P9_{geography_subset}',
        dvector=rebalanced_pop,
        dvector_dimension='population',
        detailed_logs=True,
        output_level=OutputLevel.INTERMEDIATE
    )
    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f'Output P9_{geography_subset}_VALIDATION.csv',
        float_format='%.5f', index=False
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE,
        file=f'Output P9_{geography_subset}_VALIDATION.xlsx',
        dfs=differences
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        hh_age_gender_2021_target, adjusted_pop
    )

    # --- Step 10 --- #
    LOGGER.info('--- Step 10 ---')

    # applying IPF (adjusting totals to match P9 outputs)
    LOGGER.info('Applying IPF for independent population targets')
    # calculate adjustment factor for 2021 population at a total level and apply to the IPF target
    # TODO This is because children ages are small in adjusted_pop and adding as explicit target means these are matched when they probs shouldnt be
    population_adjustments = []
    for target in population_adjustment:
        adjustment_factor = rebalanced_pop.total / target.total
        population_adjustments.append(target * adjustment_factor)

    ipfed_pop, summary, differences = data_processing.apply_ipf(
        seed_data=rebalanced_pop,
        target_dvectors=population_adjustments,
        cache_folder=constants.CACHE_FOLDER
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P10_{geography_subset}',
        dvector=ipfed_pop,
        dvector_dimension='population',
        detailed_logs=True,
        output_level=OutputLevel.INTERMEDIATE
    )
    summary.to_csv(
        OUTPUT_DIR / OutputLevel.INTERMEDIATE / f'Output P10_{geography_subset}_VALIDATION.csv',
        float_format='%.5f', index=False
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.INTERMEDIATE,
        file=f'Output P10_{geography_subset}_VALIDATION.xlsx',
        dfs=differences
    )

    # clear data at the end of the loop
    data_processing.clear_dvectors(
        rebalanced_pop, *population_adjustment
    )

    return BaseYearPopulationData(
        population=ipfed_pop, households=rebalanced_hh,
        unoccupied_factor=unoccupied_factor
    )

def rebase(
        configuration: dict,
        base_year_data: BaseYearPopulationData,
        geography_subset: str
) -> Tuple[DVector]:

    # read in the household validation data from the config file
    LOGGER.info(f'Importing household rebase data from config file')
    household_growth = data_processing.read_dvector_from_config(
        config=configuration,
        data_block='household_adjustment_data',
        key='rebase_data',
        geography_subset=geography_subset
    )[0]

    # read in the population adjustment data from the config file
    LOGGER.info(f'Importing population rebase data from config file')
    population_adjustment = data_processing.read_dvector_from_config(
        config=configuration,
        data_block='population_adjustment_data',
        key='rebase_data',
        geography_subset=geography_subset
    )
    # bring in segmentations to maintain from the 2021 build datasets
    # try looking for `rebase_segments_to_maintain` key and log if not provided
    rebase_segments_to_maintain = configuration['population_adjustment_data'].get(
        'rebase_segments_to_maintain', []
    )

    # loop through the supplied segmentations, aggregating the 2021 population
    # data to each of the segmentations provided and deriving a monovariate
    # target for the IPF and adding it to the start of the rebase targets
    added_targets = []
    for segmentation in rebase_segments_to_maintain:
        if isinstance(segmentation, list):
            target = base_year_data.population.aggregate(segs=segmentation)
        else:
            target = base_year_data.population.aggregate(segs=[segmentation])
        added_targets.append(target)

    population_adjustment_targets = added_targets + population_adjustment

    # --- Step 11 --- #
    LOGGER.info('--- Step 11 ---')
    LOGGER.info('Rebasing population to 2023')
    rebased_pop, summary, differences = data_processing.apply_ipf(
        seed_data=base_year_data.population,
        target_dvectors=population_adjustment_targets,
        cache_folder=constants.CACHE_FOLDER,
        # todo change
        target_dvector=population_adjustment[0],
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P11_{geography_subset}',
        dvector=rebased_pop,
        dvector_dimension='population',
        output_level=OutputLevel.FINAL
    )
    (OUTPUT_DIR / OutputLevel.ASSURANCE).mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUTPUT_DIR / OutputLevel.ASSURANCE / f'Output P11_{geography_subset}_VALIDATION.csv',
        float_format='%.5f', index=False
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.ASSURANCE,
        file=f'Output P11_{geography_subset}_VALIDATION.xlsx',
        dfs=differences
    )

    # rebased_pop = DVector.load(rf"F:\Deliverables\Land-Use\241218_Population\02_Final Outputs\Output P11_{geography_subset}.hdf")

    # --- Step 12 --- #
    LOGGER.info('--- Step 12 ---')
    LOGGER.info('Rebasing households to 2023')

    # get 2021 average occupancies by zone
    census_occupancy = (
            base_year_data.population.add_segments(['total']).aggregate(['total']) /
            base_year_data.households.add_segments(['total']).aggregate(['total'])
    )

    # use this to derive approximate 2023 households totals by LSOA
    rebase_hh_approx = rebased_pop.add_segments(['total']).aggregate(['total']) / census_occupancy

    # calculate total growth in 2021 to 2023 households by district based on this
    # approximation
    rebased_hh_total = rebase_hh_approx.aggregate(["total"]).translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER
    )
    census_hh_total = base_year_data.households.add_segments(['total']).aggregate(["total"]).translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LAD2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER
    )
    derived_growth = rebased_hh_total / census_hh_total

    # compare this derived growth to the actual growth to get district based control factors
    control_factors = household_growth / derived_growth

    # translate the control factors back to LSOA
    control_factors_lsoa = control_factors.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    # apply these control factors to the approximate 2023 households
    rebase_hh = rebase_hh_approx * control_factors_lsoa

    # apply household type splits from the 2021 data
    rebase_hh = data_processing.apply_proportions(
        source_dvector=base_year_data.households.add_segments(['total']),
        apply_to=rebase_hh
    ).aggregate(base_year_data.households.segmentation.names)

    # derive household total targets based on growth applied to 2021 households
    # at district level
    rebase_household_target = census_hh_total * household_growth

    # derive household constraints from population data (required occupancies)
    household_adjustment_targets = data_processing.derive_household_occupancy_targets(
        population_dvector=rebased_pop
    )
    rebased_households, summary, differences = data_processing.apply_ipf(
        seed_data=rebase_hh,
        target_dvectors=list(household_adjustment_targets),
        cache_folder=constants.CACHE_FOLDER,
    )

    # save output to hdf and csvs for checking
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P12_{geography_subset}',
        dvector=rebased_households,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )
    (OUTPUT_DIR / OutputLevel.ASSURANCE).mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUTPUT_DIR / OutputLevel.ASSURANCE / f'Output P12_{geography_subset}_VALIDATION.csv',
        float_format='%.5f', index=False
    )
    data_processing.write_to_excel(
        output_folder=OUTPUT_DIR / OutputLevel.ASSURANCE,
        file=f'Output P12_{geography_subset}_VALIDATION.xlsx',
        dfs=differences
    )

    # --- Step 13 --- #
    LOGGER.info('--- Step 13 ---')
    LOGGER.info('Applying occupancy rules to households')

    LOGGER.info('Set households to 0 where there is 0 population')
    # set households to zero where there is no population in a given ['adult', 'children'] segment and zone
    rebased_households = data_processing.mask_zero_population(
        population_dvector=rebased_pop, household_dvector=rebased_households
    )

    # save output to hdf
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P13.1_{geography_subset}',
        dvector=rebased_households,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )

    LOGGER.info(
        f'Cap maximum occupancies based on the '
        f'{float(configuration["occupancy_cap_percentile"])}th percentile'
    )
    # get max_percentile cap by adult and children combination for all zones in the
    # data and cap resulting occupancies
    rebased_households = data_processing.cap_household_occupancies(
        population_dvector=rebased_pop, household_dvector=rebased_households,
        aggregate_zone_system_name=f'RGN2021-{geography_subset}',
        current_zone_system_name=f'LSOA2021-{geography_subset}',
        percentile=float(config["occupancy_cap_percentile"])
    )

    # save output to hdf
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P13.2_{geography_subset}',
        dvector=rebased_households,
        dvector_dimension='households',
        output_level=OutputLevel.INTERMEDIATE
    )

    LOGGER.info(f'Cap minimum occupancies based on the household type')
    # get resulting occupancies by adults and children
    resulting_occupancies = (
            data_processing.filter_to_adults(rebased_pop).aggregate(['adults', 'children', 'ns_sec', 'accom_h'])
            / rebased_households.aggregate(['adults', 'children', 'ns_sec', 'accom_h'])
    )

    # get lower caps by adult and children combinations
    region_code = constants.KNOWN_GEOGRAPHIES.get(f'RGN2021-{geography_subset}').zone_ids[0]
    lower_caps = resulting_occupancies.data.min(axis=1).rename(region_code).to_frame()
    lower_caps[region_code] = 0
    min_caps = {
        (1, 2): 1,
        (2, 2): 2,
        (3, 1): 3,
        (3, 2): 3
    }
    for (adults, children), min_cap in min_caps.items():
        lower_caps[
            (lower_caps.index.get_level_values('adults') == adults) &
            (lower_caps.index.get_level_values('children') == children)
        ] = min_cap

    # convert the caps to DVector format at region level
    lower_caps = data_processing.create_dvector_from_data(
        dvector_data=lower_caps,
        geographical_level='RGN2021',
        input_segments=['adults', 'children', 'ns_sec', 'accom_h'],
        geography_subset=geography_subset
    )
    # convert these percentiles to LSOA
    lower_caps = lower_caps.translate_zoning(
        new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'LSOA2021-{geography_subset}'),
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.NO_WEIGHT,
        check_totals=False
    )

    # calculate adjustment factors for zones which have occupancy over the max_percentile
    control_factors = lower_caps / resulting_occupancies
    control_factors._data = control_factors._data.replace(np.inf, np.nan).fillna(1)
    control_factors._data = control_factors._data.where(control_factors._data > 1, 1)

    # apply these factors back to the households, to increase the number of
    # households to decrease occupancy
    rebased_households = rebased_households / control_factors

    # save output to hdf
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P13.3_{geography_subset}',
        dvector=rebased_households,
        dvector_dimension='households',
        output_level=OutputLevel.FINAL
    )

    # --- Step 14 --- #
    LOGGER.info('--- Step 14 ---')
    LOGGER.info('Getting occupied and unoccupied dwellings')

    occupied_households = rebased_households.aggregate(['accom_h'])
    unoccupied_households = occupied_households * base_year_data.unoccupied_factor

    # save outputs to hdf
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P14.1_{geography_subset}',
        dvector=occupied_households,
        dvector_dimension='households',
        output_level=OutputLevel.FINAL
    )
    data_processing.save_output(
        output_folder=OUTPUT_DIR,
        output_reference=f'Output P14.2_{geography_subset}',
        dvector=unoccupied_households,
        dvector_dimension='households',
        output_level=OutputLevel.FINAL
    )

    return rebased_pop, rebased_households


# TODO: expand on the documentation here
parser = ArgumentParser('Land-Use base population command line runner')
parser.add_argument('config_file', type=Path)
args = parser.parse_args()

# load configuration file
with open(args.config_file, 'r') as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

# Get output directory for intermediate outputs from config file
OUTPUT_DIR = Path(config['output_directory'])
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Define whether to output intermediate outputs, recommended to not output loads if debugging
generate_summary_outputs = bool(config['output_intermediate_outputs'])

# define whether to run scotland population only (requires outputs of all the
# separate GOR population builds)
run_scotland_only = bool(config['run_scotland_only'])

# Set up logger (different name if running scotland only)
if not run_scotland_only:
    LOGGER = lu_logging.configure_logger(
        output_dir=OUTPUT_DIR / OutputLevel.SUPPORTING,
        log_name='population'
    )

    # copy config file for traceability
    shutil.copy(
        src=args.config_file,
        dst=OUTPUT_DIR / OutputLevel.SUPPORTING / args.config_file.name
    )

    # loop through GORs to save memory issues further down the line
    for GOR in constants.GORS:
        # Try and load in base year data
        try:
            base_data = BaseYearPopulationData.from_folder(
                Path(config['base_year_folder']), identifier=f'_{GOR}'
            )
        except (KeyError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                LOGGER.warning('Base year data could not be found. Attempting to re-process')
            base_data = process_base(config, geography_subset=GOR)

        rebase(config, base_data, geography_subset=GOR)

        # trying to delete data to save memory to hopefully allow scotland
        # processing to run subsequently, currently crashes with memory error
        base_data = None

        LOGGER.info(f'*****COMPLETED PROCESSING FOR {GOR}*****')

# SCOTLAND-SPECIFIC PROCESSING
else:
    LOGGER = lu_logging.configure_logger(
        output_dir=OUTPUT_DIR / OutputLevel.SUPPORTING,
        log_name='scotland_population'
    )

LOGGER.info('Applying regional profiles to Scotland population data')
area_type_agg = []
for gor in config['scotland_donor_regions']:
    LOGGER.debug(f'Re-reading P11 for {gor}')
    final_pop = DVector.load(
        OUTPUT_DIR / OutputLevel.FINAL / f'Output P11_{gor}.hdf'
    )
    area_type_agg.append(
        final_pop.translate_zoning(constants.TFN_AT_AGG_ZONING_SYSTEM, cache_path=constants.CACHE_FOLDER)
    )

LOGGER.debug('Disaggregating area types to Scotland')
# Accumulate England totals at area type, then disaggregate to Scotland zoning
england_totals = reduce(lambda x, y: x+y, area_type_agg)

# Clear out the individual DVectors for England (in case of memory issues)
data_processing.clear_dvectors(*area_type_agg)

england_totals_scotland_zoning = england_totals.translate_zoning(
    constants.SCOTLAND_DZONE_ZONING_SYSTEM, cache_path=constants.CACHE_FOLDER
)

# Read in the Scotland data, and then apply proportions
scotland_population = data_processing.read_dvector_from_config(
    config=config,
    data_block='base_data',
    key='scotland_population'
)

scotland_hydrated = data_processing.apply_proportions(
    source_dvector=england_totals_scotland_zoning, apply_to=scotland_population
)

# get rid of any additional segmentations that aren't required
LOGGER.debug('Removing any superfluous segments from Scotland data')
scotland_hydrated = scotland_hydrated.aggregate(
    england_totals.segmentation
)

# save output to hdf
data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P11_Scotland',
    dvector=scotland_hydrated,
    dvector_dimension='population',
    detailed_logs=True,
    output_level=OutputLevel.FINAL
)

# collapse segmentation to household-specific segmentations (i.e. remove population segmentation)
aggregated_pop = scotland_hydrated.aggregate(
    segs=['accom_h', 'ns_sec', 'adults', 'children', 'car_availability', 'adult_nssec']
)

# # Clear out the massive DVector for scotland (in case of memory issues)
# data_processing.clear_dvectors(scotland_hydrated)

# read in the occupied and unoccupied households from the donor regions
area_type_agg_ons = []
area_type_agg_occ = []
area_type_agg_unocc = []
for gor in config['scotland_donor_regions']:
    LOGGER.debug(f'Re-reading ONS table 1 for {gor}')
    ons_table_1 = data_processing.read_dvector_from_config(
        config=config,
        data_block='base_data',
        key='ons_table_1',
        geography_subset=gor
    )
    area_type_agg_ons.append(
        ons_table_1.translate_zoning(
            constants.TFN_AT_AGG_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER
        )
    )
    LOGGER.debug(f'Re-reading occupied households for {gor}')
    occupied_households = data_processing.read_dvector_from_config(
        config=config,
        data_block='base_data',
        key='occupied_households',
        geography_subset=gor
    )

    area_type_agg_occ.append(
        occupied_households.translate_zoning(
            constants.TFN_AT_AGG_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER
        )
    )
    LOGGER.debug(f'Re-reading unoccupied households for {gor}')
    unoccupied_households = data_processing.read_dvector_from_config(
        config=config,
        data_block='base_data',
        key='unoccupied_households',
        geography_subset=gor
    )

    area_type_agg_unocc.append(
        unoccupied_households.translate_zoning(
            constants.TFN_AT_AGG_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER
        )
    )

LOGGER.debug('Disaggregating area types to Scotland')
# Accumulate England totals at area type
england_ons_totals = reduce(lambda x, y: x+y, area_type_agg_ons)
england_occ_totals = reduce(lambda x, y: x+y, area_type_agg_occ)
england_unocc_totals = reduce(lambda x, y: x+y, area_type_agg_unocc)

# Clear out the individual DVectors for England (in case of memory issues)
data_processing.clear_dvectors(
    *area_type_agg_ons, *area_type_agg_occ, *area_type_agg_unocc
)

# calculate average occupancy, as in base_population.py
england_average_occupancy = (england_ons_totals / england_occ_totals)
# replace infinities with nans for infilling
# this is where the occupied_households value is zero for a dwelling type and zone,
# but the ons_table_1 has non-zero population. E.g. LSOA E01007423 in GOR = 'YH'
# caravans and mobile homes, the occupied households = 0 but ons_table_1 population = 4
england_average_occupancy._data = england_average_occupancy._data.replace(np.inf, np.nan)
# infill missing occupancies with average value of other properties in the zone
# i.e. based on column
england_average_occupancy._data = england_average_occupancy._data.fillna(
    england_average_occupancy._data.mean(axis=0), axis=0
)

# convert occupancy factors to scotland zoning
england_occupancy_scotland_zoning = england_average_occupancy.translate_zoning(
    constants.SCOTLAND_DZONE_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.NO_WEIGHT,
    check_totals=False
)

# divide scottish population by average occupancy to get households
rebase_hh = aggregated_pop / england_occupancy_scotland_zoning

# derive household constraints from population data (required occupancies)
household_adjustment_targets = data_processing.derive_household_occupancy_targets(
    population_dvector=scotland_hydrated
)
rebased_households, summary, differences = data_processing.apply_ipf(
    seed_data=rebase_hh,
    target_dvectors=list(household_adjustment_targets),
    cache_folder=constants.CACHE_FOLDER,
)
# save output to hdf and csvs for checking
data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P12_Scotland',
    dvector=rebased_households,
    dvector_dimension='households',
    output_level=OutputLevel.INTERMEDIATE
)
(OUTPUT_DIR / OutputLevel.ASSURANCE).mkdir(parents=True, exist_ok=True)
summary.to_csv(
    OUTPUT_DIR / OutputLevel.ASSURANCE / f'Output P12_Scotland_VALIDATION.csv',
    float_format='%.5f', index=False
)
data_processing.write_to_excel(
    output_folder=OUTPUT_DIR / OutputLevel.ASSURANCE,
    file=f'Output P12_Scotland_VALIDATION.xlsx',
    dfs=differences
)

LOGGER.info('Set households to 0 where there is 0 population')
# set households to zero where there is no population in a given ['adult', 'children'] segment and zone
rebased_households = data_processing.mask_zero_population(
    population_dvector=scotland_hydrated, household_dvector=rebased_households,
    household_segments=('adults', 'children', 'ns_sec', 'accom_h')
)

# save output to hdf
data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P13.1_Scotland',
    dvector=rebased_households,
    dvector_dimension='households',
    output_level=OutputLevel.INTERMEDIATE
)

LOGGER.info(
        f'Cap maximum occupancies based on the '
        f'{float(config["occupancy_cap_percentile"])}th percentile'
    )
# get max_percentile cap by adult and children combination for all zones in the
# data and cap resulting occupancies
rebased_households = data_processing.cap_household_occupancies(
    population_dvector=scotland_hydrated, household_dvector=rebased_households,
    aggregate_zone_system_name='SCOTLANDRGN',
    current_zone_system_name='DZ2011',
    percentile=float(config["occupancy_cap_percentile"])
)

data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P13.2_Scotland',
    dvector=rebased_households,
    dvector_dimension='households',
    output_level=OutputLevel.INTERMEDIATE
)

LOGGER.info(f'Cap minimum occupancies based on the household type')
# get resulting occupancies by adults and children
resulting_occupancies = (
        data_processing.filter_to_adults(scotland_hydrated).aggregate(['adults', 'children', 'ns_sec', 'accom_h'])
        / rebased_households.aggregate(['adults', 'children', 'ns_sec', 'accom_h'])
)

# get lower caps by adult and children combinations
region_code = constants.KNOWN_GEOGRAPHIES.get(f'SCOTLANDRGN').zone_ids[0]
lower_caps = resulting_occupancies.data.min(axis=1).rename(region_code).to_frame()
lower_caps[region_code] = 0
min_caps = {
    (1, 2): 1,
    (2, 2): 2,
    (3, 1): 3,
    (3, 2): 3
}
for (adults, children), min_cap in min_caps.items():
    lower_caps[
        (lower_caps.index.get_level_values('adults') == adults) &
        (lower_caps.index.get_level_values('children') == children)
    ] = min_cap

# convert the caps to DVector format at region level
lower_caps = data_processing.create_dvector_from_data(
    dvector_data=lower_caps,
    geographical_level='SCOTLANDRGN',
    input_segments=['adults', 'children', 'ns_sec', 'accom_h']
)
# convert these percentiles to LSOA
lower_caps = lower_caps.translate_zoning(
    new_zoning=constants.KNOWN_GEOGRAPHIES.get(f'DZ2011'),
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.NO_WEIGHT,
    check_totals=False
)

# calculate adjustment factors for zones which have occupancy over the max_percentile
control_factors = lower_caps / resulting_occupancies
control_factors._data = control_factors._data.replace(np.inf, np.nan).fillna(1)
control_factors._data = control_factors._data.where(control_factors._data > 1, 1)

# apply these factors back to the households, to increase the number of
# households to decrease occupancy
rebased_households = rebased_households / control_factors

data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P13.3_Scotland',
    dvector=rebased_households,
    dvector_dimension='households',
    detailed_logs=True,
    output_level=OutputLevel.FINAL
)

LOGGER.info('Getting occupied and unoccupied dwellings')

occupied_households = rebased_households.aggregate(['accom_h'])

# calculate unoccupied households
empty_proportion = (england_unocc_totals / england_occ_totals)
empty_proportion._data = empty_proportion._data.replace(np.inf, np.nan)
# infill missing occupancies with average value of other properties in the zone
# i.e. based on column
empty_proportion._data = empty_proportion._data.fillna(
    empty_proportion._data.mean(axis=0), axis=0
)
unoccupied_households = occupied_households * empty_proportion.translate_zoning(
    constants.SCOTLAND_DZONE_ZONING_SYSTEM,
    cache_path=constants.CACHE_FOLDER,
    weighting=TranslationWeighting.NO_WEIGHT,
    check_totals=False
)

# save outputs to hdf
data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P14.1_Scotland',
    dvector=occupied_households,
    dvector_dimension='households',
    output_level=OutputLevel.FINAL
)
data_processing.save_output(
    output_folder=OUTPUT_DIR,
    output_reference=f'Output P14.2_Scotland',
    dvector=unoccupied_households,
    dvector_dimension='households',
    output_level=OutputLevel.FINAL
)
