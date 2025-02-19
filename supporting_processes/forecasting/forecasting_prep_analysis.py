"""
Various functions that have been used to extract/format datasets for ONS vs NTEM
Analysis spreadsheet saved here: F:\Working\Land-Use\FORECASTING_prep\Forecast investigations spreadsheet
"""

from pathlib import Path
import land_use.preprocessing as pp
import pandas as pd
from land_use import constants, data_processing
from caf.base.zoning import TranslationWeighting
from caf.base import DVector

OUTPUT_DIR = Path(r'F:\Working\Land-Use\FORECASTING_PREP\01_Intermediate Files')
NTEM_INPUT_DIR = Path(r'I:\Data\NTEM\MG_NorCOM_review_121324')
BASE_EMP_OUTPUTS_DIR = Path(r'F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs')


def ntem_ca_to_dvector():
    """ Function to reformat the NTEM car availability data for forecasting into DVector format """

    seg_names = 'car_availability'

    for scenario in ['Core', 'High', 'Low']:
        file_path = NTEM_INPUT_DIR / fr'ntem_8.0_{scenario}_ca_data.csv'

        for year in ['2023', '2028', '2033', '2038', '2043', '2048']:
            ntem = pd.read_csv(file_path, usecols=['msoa_zone_id', 'car_ownership', year])

            # sum together car_ownership NTEM 2 cars and 3+ cars categories
            segmentation_mapping = {'no_car': 1, '1_car': 2, '2_cars': 3, '3+_cars': 3}
            ntem[seg_names] = ntem['car_ownership'].map(segmentation_mapping)
            ntem_grouped = ntem.groupby(['msoa_zone_id', seg_names], as_index=False).sum().sort_values(
                by=['msoa_zone_id', seg_names], ascending=True).drop(columns='car_ownership')

            # get this long format dataframe into a wide format for DVector
            ntem_wide = pp.pivot_to_dvector(
                data=ntem_grouped,
                zoning_column='msoa_zone_id',
                index_cols=[seg_names],
                value_column=year,
            )

            pp.save_preprocessed_hdf(source_file_path=file_path, df=ntem_wide, multiple_output_ref=year)


def emp_outputs_base_rgn_output():
    """ Function to summarise the Base employment outputs by segmentation, translated to regions """

    final_dfs = []
    for output in ['Output E4', 'Output E4_2', 'Output E4_3', 'Output E5', 'Output E6']:
        # read in the employment output hdf
        dv = DVector.load(BASE_EMP_OUTPUTS_DIR / fr'{output}.hdf')
        # translate output into Region zoning
        dv_translated = dv.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False
        )

        # aggregate segmentation
        dv2 = dv_translated.aggregate(['sic_1_digit'])
        dv3 = dv_translated.aggregate(['soc'])
        dv4 = dv_translated.aggregate(['sic_2_digit'])
        # stack columns and format
        dv2 = dv2.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv2['segmentation'] = 'sic_1_digit'
        dv2['filename'] = f'{output}'
        dv2['output code'] = f'{output}'
        dv2 = dv2[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]

        dv3 = dv3.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv3['segmentation'] = 'soc'
        dv3['filename'] = f'{output}'
        dv3['output code'] = f'{output}'
        dv3 = dv3[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]

        dv4 = dv4.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv4['segmentation'] = 'sic_2_digit'
        dv4['filename'] = f'{output}'
        dv4['output code'] = f'{output}'
        dv4 = dv4[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]

        final_dfs.append(dv2)
        final_dfs.append(dv3)
        final_dfs.append(dv4)

        if output == 'Output E5' or output == 'Output E6':
            dv5 = dv_translated.aggregate(['sic_4_digit'])
            dv5 = dv5.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
            dv5['segmentation'] = 'sic_4_digit'
            dv5['filename'] = f'{output}'
            dv5['output code'] = f'{output}'
            dv5 = dv5[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
            final_dfs.append(dv5)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(OUTPUT_DIR / r'base_emp_totals_by_region_by_output_by_segment2.csv')


def ntem_jobs_hhs_rgns():
    """ Function to summarise and format NTEM Core planning data for jobs and households,
    translated to region zoning """

    file_path = NTEM_INPUT_DIR / 'ntem_8.0_Core_planning_data.csv'
    ntem = pd.read_csv(file_path, usecols=['msoa_zone_id', 'population', '2023'])
    ntem_jobs = ntem[ntem['population'] == 'jobs'].drop(columns=['population'])
    ntem_jobs['total'] = 1
    ntem_hhs = ntem[ntem['population'] == 'HHs'].drop(columns=['population'])
    ntem_hhs['total'] = 1
    ntem_workers = ntem[ntem['population'] == 'workers'].drop(columns=['population'])
    ntem_workers['total'] = 1

    # get this long format dataframe into a wide format for DVector
    ntem_wide_jobs = pp.pivot_to_dvector(
        data=ntem_jobs,
        zoning_column='msoa_zone_id',
        index_cols=['total'],
        value_column='2023'
    )

    ntem_wide_hhs = pp.pivot_to_dvector(
        data=ntem_hhs,
        zoning_column='msoa_zone_id',
        index_cols=['total'],
        value_column='2023'
    )

    ntem_wide_workers = pp.pivot_to_dvector(
        data=ntem_workers,
        zoning_column='msoa_zone_id',
        index_cols=['total'],
        value_column='2023'
    )

    pp.save_preprocessed_hdf(source_file_path=file_path, df=ntem_wide_jobs, multiple_output_ref='jobs')
    pp.save_preprocessed_hdf(source_file_path=file_path, df=ntem_wide_hhs, multiple_output_ref='hhs')
    pp.save_preprocessed_hdf(source_file_path=file_path, df=ntem_wide_workers, multiple_output_ref='workers')

    # read the hdfs back in as DVectors, ready for zone conversion
    ntem_jobs_dv = data_processing.read_dvector_data(
        file_path=NTEM_INPUT_DIR / r'preprocessing\ntem_8.0_Core_planning_data_jobs.hdf',
        geographical_level='MSOA2011+SCOTLAND-INT-ZONE',
        input_segments=['total'])

    ntem_hhs_dv = data_processing.read_dvector_data(
        file_path=NTEM_INPUT_DIR / r'preprocessing\ntem_8.0_Core_planning_data_hhs.hdf',
        geographical_level='MSOA2011+SCOTLAND-INT-ZONE',
        input_segments=['total'])

    ntem_workers_dv = data_processing.read_dvector_data(
        file_path=NTEM_INPUT_DIR / r'preprocessing\ntem_8.0_Core_planning_data_workers.hdf',
        geographical_level='MSOA2011+SCOTLAND-INT-ZONE',
        input_segments=['total'])

    # translate zoning
    ntem_jobs_translated = ntem_jobs_dv.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )
    ntem_hhs_translated = ntem_hhs_dv.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )

    ntem_workers_translated = ntem_workers_dv.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )

    ntem_jobs_translated.save(OUTPUT_DIR / 'ntem_jobs_2023_regions.hdf')
    ntem_jobs_translated.data.to_csv(OUTPUT_DIR / 'ntem_jobs_2023_regions.csv')
    ntem_hhs_translated.save(OUTPUT_DIR / 'ntem_hhs_2023_regions.hdf')
    ntem_hhs_translated.data.to_csv(OUTPUT_DIR / 'ntem_hhs_2023_regions.csv')
    ntem_workers_translated.save(OUTPUT_DIR / 'ntem_workers_2023_regions.hdf')
    ntem_workers_translated.data.to_csv(OUTPUT_DIR / 'ntem_workers_2023_regions.csv')


def ntem_cas_totals():
    """ Function to extract and summarise the NTEM CAS, totals for each category """

    final_df = []
    for cas in ['Core', 'Low', 'High']:
        # planning data files
        input_file = NTEM_INPUT_DIR / fr'ntem_8.0_{cas}_planning_data.csv'

        df = pd.read_csv(input_file)
        df = df.groupby('population', as_index=False)[['2023', '2028', '2033', '2038', '2043', '2048']].sum()
        df = df.rename(columns={'population': 'category'})
        df['cas'] = cas
        final_df.append(df)

        # car availability files
        input_file = NTEM_INPUT_DIR / fr'ntem_8.0_{cas}_ca_data.csv'
        df = pd.read_csv(input_file)
        segmentation_mapping = {'no_car': 'no_car', '1_car': '1_car', '2_cars': '2+_car', '3+_cars': '2+_car'}
        df['car_availability'] = df['car_ownership'].map(segmentation_mapping)
        df = df.groupby('car_availability', as_index=False)[['2023', '2028', '2033', '2038', '2043', '2048']].sum()
        df = df.rename(columns={'car_availability': 'category'})
        df['cas'] = cas
        final_df.append(df)

    final_output = pd.concat(final_df)
    final_output = final_output[['category', 'cas', '2023', '2028', '2033', '2038', '2043', '2048']]
    final_output.to_csv(OUTPUT_DIR / 'ntem_cas_totals_by_category.csv')


def ntem_cas_totals_rgns_all():
    """ Function to extract and summarise the NTEM CAS, totals for each category, translated to region zoning """

    # PLANNING DATA
    for cas in ['Low', 'High']:
        # planning data files
        input_file = NTEM_INPUT_DIR / fr'ntem_8.0_{cas}_planning_data.csv'

        for year in ['2023', '2028', '2033', '2038', '2043', '2048']:
            for cat in ['under16', '16-74', '75+', 'HHs', 'jobs', 'workers']:
                df = pd.read_csv(input_file, usecols=['msoa_zone_id', 'population', year])
                df = df[df['population'] == cat].drop(columns=['population'])
                df['total'] = 1

                # get this long format dataframe into a wide format for DVector
                df = pp.pivot_to_dvector(
                    data=df,
                    zoning_column='msoa_zone_id',
                    index_cols=['total'],
                    value_column=year
                )
                pp.save_preprocessed_hdf(source_file_path=input_file, df=df, multiple_output_ref=f'{year}_{cat}')
    # CAR AVAILABILITY
    for cas in ['Low', 'Core', 'High']:
        # planning data files
        input_file = NTEM_INPUT_DIR / fr'ntem_8.0_{cas}_ca_data.csv'

        for year in ['2023', '2028', '2033', '2038', '2043', '2048']:
            df = pd.read_csv(input_file, usecols=['msoa_zone_id', 'car_ownership', year])
            ca_mapping = {'no_car': 'no_car', '1_car': '1_car', '2_cars': '2+_car', '3+_cars': '2+_car'}
            df['car_availability'] = df['car_ownership'].map(ca_mapping)
            df = df.groupby(['msoa_zone_id', 'car_availability'], as_index=False)[[f'{year}']].sum()
            df = df.rename(columns={'car_availability': 'category'})

            for cat in ['no_car', '1_car', '2+_car']:
                df_cat = df[df['category'] == cat].drop(columns=['category'])
                df_cat['total'] = 1

                # get this long format dataframe into a wide format for DVector
                df_long = pp.pivot_to_dvector(
                    data=df_cat,
                    zoning_column='msoa_zone_id',
                    index_cols=['total'],
                    value_column=f'{year}'
                )

                pp.save_preprocessed_hdf(source_file_path=input_file, df=df_long, multiple_output_ref=f'{year}_{cat}')

    output_dfs = []
    for cas in ['Low', 'Core', 'High']:
        # read back in the data as hdfs and translate zoning
        for year in ['2023', '2028', '2033', '2038', '2043', '2048']:
            for cat in ['under16', '16-74', '75+', 'HHs', 'jobs', 'workers', 'no_car', '1_car', '2+_car']:
                # read the hdfs back in as DVectors, ready for zone conversion
                if cat == 'no_car' or cat == '1_car' or cat == '2+_car':
                    data_type = 'ca'
                else:
                    data_type = 'planning'
                dv = data_processing.read_dvector_data(
                    file_path=NTEM_INPUT_DIR / fr'preprocessing\ntem_8.0_{cas}_{data_type}_data_{year}_{cat}.hdf',
                    geographical_level='MSOA2011+SCOTLAND-INT-ZONE',
                    input_segments=['total'])

                # print segmentation
                print(dv.segmentation.names)
                # translate zoning
                dv_translated = dv.translate_zoning(
                    new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                    cache_path=constants.CACHE_FOLDER,
                    weighting=TranslationWeighting.SPATIAL,
                    check_totals=False
                )

                output_df = dv_translated.data
                output_df = output_df.T.rename(columns={1: f'{cas}_{year}_{cat}'})

                output_df['category'] = output_df.columns[0].split('_')[2]
                output_df['cas'] = output_df.columns[0].split('_')[0]
                output_df = output_df.reset_index().rename(columns={'msoa_zone_id': 'region', f'{cas}_{year}_{cat}': f'{year}'})
                output_df = output_df[['region', 'category', 'cas', f'{year}']]

                # map the region codes
                region_mapping = {'W92000004': 'Wales', 'E12000008': 'SE', 'E12000004': 'EM',
                                  'E12000005': 'WM', 'E12000002': 'NW', 'E12000009': 'SW',
                                  'E12000007': 'Lon', 'E12000003': 'YH', 'E12000001': 'NE',
                                  'E12000006': 'EoE', 'S92000003': 'Scotland'}
                output_df['region'] = output_df['region'].map(region_mapping)

                output_dfs.append(output_df)

    final_output = pd.concat(output_dfs).fillna(0)
    final_output = final_output.replace({'no': 'no_car', '1': '1_car', '2+': '2+_car'})
    final_output = final_output.groupby(by=['region', 'category', 'cas'], as_index=False)[['2023', '2028', '2033', '2038', '2043', '2048']].sum()
    final_output = final_output[['category', 'region', 'cas', '2023', '2028', '2033', '2038', '2043', '2048']]
    final_output.to_csv(OUTPUT_DIR / 'ntem_cas_totals_by_category_by_region.csv')


def ntem_cas_totals_rgns_all_agg():
    """ Function to extract and summarise the NTEM CAS, aggregating the existing outputs
    into 'England' and The 'North' """

    new_df = pd.read_csv(OUTPUT_DIR / 'ntem_cas_totals_by_category_by_region.csv')
    england_mapping = {'EM': 'England', 'EoE': 'England', 'Lon': 'England', 'NE': 'England',
                       'NW': 'England', 'SE': 'England', 'SW': 'England', 'WM': 'England',
                       'YH': 'England', 'Wales': 'Wales', 'Scotland': 'Scotland'}
    the_north_mapping = {'EM': 'other', 'EoE': 'other', 'Lon': 'other', 'NE': 'The North',
                         'NW': 'The North', 'SE': 'other', 'SW': 'other', 'WM': 'other',
                         'YH': 'The North', 'Wales': 'other', 'Scotland': 'other'}
    new_df['england'] = new_df['region'].map(england_mapping)
    new_df['the_north'] = new_df['region'].map(the_north_mapping)

    england = new_df.groupby(by=['england', 'category', 'cas'], as_index=False)[[
        '2023', '2028', '2033', '2038', '2043', '2048']].sum()
    england = england[['category', 'england', 'cas', '2023', '2028', '2033', '2038', '2043', '2048']]
    the_north = new_df.groupby(by=['the_north', 'category', 'cas'], as_index=False)[[
        '2023', '2028', '2033', '2038', '2043', '2048']].sum()
    the_north = the_north[['category', 'the_north', 'cas', '2023', '2028', '2033', '2038', '2043', '2048']]

    england.to_csv(OUTPUT_DIR / 'ntem_cas_totals_by_category_england.csv')
    the_north.to_csv(OUTPUT_DIR / 'ntem_cas_totals_by_category_thenorth.csv')


def ntem_core_totals_by_la():
    """ Summarise NTEM Core totals, translated into LAD zoning"""

    file_path = NTEM_INPUT_DIR / fr'ntem_8.0_Core_planning_data.csv'

    for year in ['2023', '2028', '2033', '2038', '2043', '2048']:
        for cat in ['under16', '16-74', '75+', 'HHs', 'jobs', 'workers']:
            ntem = pd.read_csv(file_path, usecols=['msoa_zone_id', 'population', year])
            ntem = ntem[ntem['population'] == cat].drop(columns=['population'])
            ntem['total'] = 1

            # get this long format dataframe into a wide format for DVector
            ntem_wide = pp.pivot_to_dvector(
                data=ntem,
                zoning_column='msoa_zone_id',
                index_cols=['total'],
                value_column=year
            )

            pp.save_preprocessed_hdf(source_file_path=file_path, df=ntem_wide, multiple_output_ref=f'{year}_{cat}')

    output_dfs = []
    # read back in the data as hdfs and translate zoning
    for year in ['2023', '2028', '2033', '2038', '2043', '2048']:
        for cat in ['under16', '16-74', '75+', 'HHs', 'jobs', 'workers']:
            # read the hdfs back in as DVectors, ready for zone conversion
            ntem_dv = data_processing.read_dvector_data(
                file_path=NTEM_INPUT_DIR / fr'preprocessing\ntem_8.0_Core_planning_data_{year}_{cat}.hdf',
                geographical_level='MSOA2011+SCOTLAND-INT-ZONE',
                input_segments=['total'])

            # translate zoning
            ntem_translated = ntem_dv.translate_zoning(
                new_zoning=constants.LAD_EWS_2023_ZONING_SYSTEM,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.SPATIAL,
                check_totals=False
            )
            # LAD_EWS_2023_ZONING_SYSTEM
            # LAD_EWS_ZONING_SYSTEM
            output_df = ntem_translated.data
            output_df = output_df.T.rename(columns={1: f'{year}_{cat}'})

            output_dfs.append(output_df)
            # ntem_translated.save(OUTPUT_DIR / f'ntem_years_categories_as_dvectors_LAD\ntem_{year}_{cat}_LAD.hdf')

    final_output = pd.concat(output_dfs, axis=1).reset_index().rename(columns={'msoa_zone_id': 'LAD2023+SCOTLANDLAD'})
    final_output.to_csv(OUTPUT_DIR / 'ntem_by_LAD23_by_year_by_category.csv')


def output_e6_sic_soc_rgns():
    """ Function to summarise Output E6 from Base Land Use, grouping by 'sic_1_digit' and 'soc',
    translated to region zoning """

    dv = DVector.load(BASE_EMP_OUTPUTS_DIR / 'Output E6.hdf')
    # translate output into Region zoning
    dv_translated = dv.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )

    # aggregate segmentation
    dv2 = dv_translated.aggregate(['sic_1_digit', 'soc'])
    # format for output
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    dv3 = dv2.data.rename(columns=region_mapping).reset_index()
    dv3.to_csv(OUTPUT_DIR / 'lu_base_emp_SIC_1_digit_SOC_grouped_regions_test.csv')
