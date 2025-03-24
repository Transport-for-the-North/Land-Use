from pathlib import Path

import pandas as pd

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants

POP_OUTPUT_DIR = Path(r'F:\Working\Land-Use\OUTPUTS_forecast_population')
# POP_OUTPUT_DIR = Path(r'F:\Working\Land-Use\temp_forecast_population_testing\20250321')
EMP_OUTPUT_DIR = Path(r'F:\Working\Land-Use\OUTPUTS_forecast_employment')
POP_ANALYSIS_DIR = Path(r'F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\pop')
EMP_ANALYSIS_DIR = Path(r'F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\emp')


pop_years_to_extract = [2033, 2038, 2043, 2048, 2053]
emp_years_to_extract = [2033, 2038, 2043, 2048, 2053]


def summarise_population_outputs(output_file_name: str):
    # Create a summary table of the output hdfs from the main process
    final_dfs = []
    for year in pop_years_to_extract:
        for rgn in constants.GORS + ['Scotland']:
            print(f'Summarising for {year}, {rgn}')
            dv = DVector.load(Path(POP_OUTPUT_DIR / fr'01_Intermediate Files\Population_age_g_soc_'
                                                    fr'{rgn}_{year}.hdf'))
            dv_translated = dv.translate_zoning(
                new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.SPATIAL,
                check_totals=False
            )
            # aggregate segmentation
            dv1 = dv_translated.aggregate(['age_ntem'])
            dv2 = dv_translated.aggregate(['g'])
            dv3 = dv_translated.aggregate(['age_ntem', 'g'])
            dv4 = dv_translated.aggregate(['soc'])
            # dv5 = dv_translated.aggregate(['children'])

            # stack columns and format
            dv1 = dv1.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
            dv1['segmentation'] = 'age_ntem'
            dv1['filename'] = 'population'
            dv1['output code'] = 'population'
            dv1 = dv1[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
            dv1['year'] = year

            # stack columns and format
            dv2 = dv2.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
            dv2['segmentation'] = 'g'
            dv2['filename'] = 'population'
            dv2['output code'] = 'population'
            dv2 = dv2[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
            dv2['year'] = year

            dv3 = dv3.data.stack().reset_index().set_axis(['age_ntem', 'g', 'region', 'value'], axis='columns')
            dv3['age_ntem'] = dv3['age_ntem'].astype(str)
            dv3['g'] = dv3['g'].astype(str)
            dv3['segment'] = dv3['age_ntem'] + '_' + dv3['g']
            dv3['segmentation'] = 'age_g'
            dv3['filename'] = 'population'
            dv3['output code'] = 'population'
            dv3 = dv3[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
            dv3['year'] = year

            dv4 = dv4.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
            dv4['segmentation'] = 'soc'
            dv4['filename'] = 'population'
            dv4['output code'] = 'population'
            dv4 = dv4[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
            dv4['year'] = year

            # dv5 = dv5.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
            # dv5['segmentation'] = 'children'
            # dv5['filename'] = 'population'
            # dv5['output code'] = 'population'
            # dv5 = dv5[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
            # dv5['year'] = year

            final_dfs.append(dv1)
            final_dfs.append(dv2)
            final_dfs.append(dv3)
            final_dfs.append(dv4)
            # final_dfs.append(dv5)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(POP_ANALYSIS_DIR / f'{output_file_name}.csv')


def summarise_population_targets_output(output_file_name: str):
    # Create a summary table of the targets output from the main process
    targets_dfs = []
    for year in pop_years_to_extract:
        for rgn in constants.GORS + ['Scotland']:
            for output in ['pop_targets', 'soc_targets']:
                print(f'Summarising for {year}, {rgn}, {output}')
                dv = DVector.load(Path(POP_OUTPUT_DIR / f'01_Intermediate Files/{output}_{rgn}_{year}.hdf'))

                if output == 'pop_targets':
                    # stack columns and format
                    dv2 = dv.data.stack().reset_index().set_axis(['age_ntem', 'g', 'soc', 'region', 'value'], axis='columns')
                    dv2['filename'] = f'{output}'
                    dv2['output code'] = f'{output}'
                    dv2['year'] = year
                    dv2 = dv2[['filename', 'year', 'output code', 'region', 'age_ntem', 'g', 'soc', 'value']]
                else:
                    dv2 = dv.data.stack().reset_index().set_axis(['soc', 'region', 'value'], axis='columns')
                    dv2['filename'] = f'{output}'
                    dv2['output code'] = f'{output}'
                    dv2['year'] = year
                    dv2 = dv2[['filename', 'year', 'output code', 'region', 'soc', 'value']]

                targets_dfs.append(dv2)

    final_output = pd.concat(targets_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(POP_ANALYSIS_DIR / f'{output_file_name}.csv')


def summarise_emp_outputs(output_file_name: str):
    # Create a summary table of the output hdfs from the main process
    final_dfs = []
    for year in emp_years_to_extract:
        print(f'Summarising for {year}')
        dv = DVector.load(Path(EMP_OUTPUT_DIR / fr'01_Intermediate Files\Output E6_SIC_1_digit_'
                                                fr'{year}.hdf'))

        dv_translated = dv.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False
        )

        # aggregate segmentation
        dv1 = dv_translated.aggregate(['sic_1_digit'])
        dv2 = dv_translated.aggregate(['sic_2_digit'])
        dv3 = dv_translated.aggregate(['sic_4_digit'])
        dv4 = dv_translated.aggregate(['soc'])

        # stack columns and format
        dv1 = dv1.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv1['segmentation'] = 'sic_1_digit'
        dv1['year'] = year
        dv1 = dv1[['year', 'segmentation', 'region', 'segment', 'value']]

        # stack columns and format
        dv2 = dv2.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv2['segmentation'] = 'sic_2_digit'
        dv2['year'] = year
        dv2 = dv2[['year', 'segmentation', 'region', 'segment', 'value']]

        # stack columns and format
        dv3 = dv3.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv3['segmentation'] = 'sic_4_digit'
        dv3['year'] = year
        dv3 = dv3[['year', 'segmentation', 'region', 'segment', 'value']]

        # stack columns and format
        dv4 = dv4.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv4['segmentation'] = 'soc'
        dv4['year'] = year
        dv4 = dv4[['year', 'segmentation', 'region', 'segment', 'value']]

        final_dfs.append(dv1)
        final_dfs.append(dv2)
        final_dfs.append(dv3)
        final_dfs.append(dv4)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(EMP_ANALYSIS_DIR / f'{output_file_name}.csv')


def summarise_emp_targets_output(output_file_name: str):
    # Create a summary table of the targets output from the main process
    targets_dfs = []
    for year in emp_years_to_extract:
        print(f'Summarising for {year}')
        dv = DVector.load(Path(EMP_OUTPUT_DIR / f'01_Intermediate Files/sic_1_digit_targets_{year}.hdf'))
        dv2 = dv.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis='columns')
        dv2['segmentation'] = 'sic_1_digit'
        dv2['year'] = year
        dv2 = dv2[['year', 'segmentation', 'region', 'segment', 'value']]

        targets_dfs.append(dv2)

    final_output = pd.concat(targets_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(EMP_ANALYSIS_DIR / f'{output_file_name}.csv')


# summarise_population_outputs(output_file_name='population_forecast_output_summary_20250321')
# summarise_population_targets_output(output_file_name='population_forecast_targets_summary_20250321')
# summarise_emp_outputs(output_file_name='employment_forecast_output_summary')
# summarise_emp_targets_output(output_file_name='employment_forecast_targets_summary')
