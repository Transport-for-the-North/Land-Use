from pathlib import Path

import pandas as pd

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants

OUTPUT_DIR = Path(r'F:\Working\Land-Use\OUTPUTS_forecast_population')

years_to_extract = [2033, 2038, 2043, 2048]


def summarise_outputs():
    # Create a summary table of the output hdfs from the main process
    final_dfs = []
    for year in years_to_extract:
        for rgn in constants.GORS:
            for output in ['p11_step3', 'p11_step6', 'p11_step9']:
                print(f'Summarising for {year}, {rgn}, {output}')
                if output == 'p11_step3':
                    dv = DVector.load(Path(OUTPUT_DIR / fr'01_Intermediate Files\Output p11_age_g_'
                                                        fr'{rgn}_{year}.hdf'))
                elif output == 'p11_step6':
                    dv = DVector.load(Path(OUTPUT_DIR / fr'01_Intermediate Files\Output p11_age_g_soc_'
                                                        fr'{rgn}_{year}.hdf'))
                else:
                    dv = DVector.load(Path(OUTPUT_DIR / fr'01_Intermediate Files\Output p11_age_g_soc_children_'
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
                dv5 = dv_translated.aggregate(['children'])

                # stack columns and format
                dv1 = dv1.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
                dv1['segmentation'] = 'age_ntem'
                dv1['filename'] = f'{output}'
                dv1['output code'] = f'{output}'
                dv1 = dv1[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
                dv1['year'] = year

                # stack columns and format
                dv2 = dv2.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
                dv2['segmentation'] = 'g'
                dv2['filename'] = f'{output}'
                dv2['output code'] = f'{output}'
                dv2 = dv2[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
                dv2['year'] = year

                dv3 = dv3.data.stack().reset_index().set_axis(['age_ntem', 'g', 'region', 'value'], axis='columns')
                dv3['age_ntem'] = dv3['age_ntem'].astype(str)
                dv3['g'] = dv3['g'].astype(str)
                dv3['segment'] = dv3['age_ntem'] + '_' + dv3['g']
                dv3['segmentation'] = 'age_g'
                dv3['filename'] = f'{output}'
                dv3['output code'] = f'{output}'
                dv3 = dv3[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
                dv3['year'] = year

                dv4 = dv4.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
                dv4['segmentation'] = 'soc'
                dv4['filename'] = f'{output}'
                dv4['output code'] = f'{output}'
                dv4 = dv4[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
                dv4['year'] = year

                dv5 = dv5.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
                dv5['segmentation'] = 'children'
                dv5['filename'] = f'{output}'
                dv5['output code'] = f'{output}'
                dv5 = dv5[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]
                dv5['year'] = year

                final_dfs.append(dv1)
                final_dfs.append(dv2)
                final_dfs.append(dv3)
                final_dfs.append(dv4)
                final_dfs.append(dv5)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(OUTPUT_DIR / r'Analysis\outputs\population_forecast_output_summary.csv')


def summarise_targets_output():
    # Create a summary table of the targets output from the main process
    targets_dfs = []
    for year in years_to_extract:
        for rgn in constants.GORS:
            for output in ['pop_targets', 'hh_children_targets']:
                print(f'Summarising for {year}, {rgn}, {output}')
                dv = DVector.load(Path(OUTPUT_DIR / f'01_Intermediate Files/{output}_{year}_{rgn}.hdf'))

                if output == 'pop_targets':
                    # stack columns and format
                    dv2 = dv.data.stack().reset_index().set_axis(['age_ntem', 'g', 'region', 'value'], axis='columns')
                    dv2['age_ntem'] = dv2['age_ntem'].astype(str)
                    dv2['g'] = dv2['g'].astype(str)
                    dv2['segment'] = dv2['age_ntem'] + '_' + dv2['g']
                    dv2['segmentation'] = 'age_ntem_g'
                    dv2['filename'] = f'{output}'
                    dv2['output code'] = f'{output}'
                    dv2['year'] = year
                    dv2 = dv2[['filename', 'year', 'segmentation', 'output code', 'region', 'segment', 'value']]
                else:
                    dv2 = dv.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis='columns')
                    dv2['segmentation'] = 'children'
                    dv2['filename'] = f'{output}'
                    dv2['output code'] = f'{output}'
                    dv2['year'] = year
                    dv2 = dv2[['filename', 'year', 'segmentation', 'output code', 'region', 'segment', 'value']]

                targets_dfs.append(dv2)

    final_output = pd.concat(targets_dfs)
    # redefine the region names
    region_mapping = {'W92000004': 'Wales', 'E12000008': 'South East', 'E12000004': 'East Midlands',
                      'E12000005': 'West Midlands', 'E12000002': 'North West', 'E12000009': 'South West',
                      'E12000007': 'London', 'E12000003': 'Yorkshire and The Humber', 'E12000001': 'North East',
                      'E12000006': 'East of England', 'S92000003': 'Scotland'}
    final_output['region'] = final_output['region'].map(region_mapping)
    final_output.to_csv(OUTPUT_DIR / r'Analysis\outputs\population_forecast_targets_summary.csv')


summarise_outputs()
summarise_targets_output()
