from pathlib import Path

import pandas as pd

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants

final_dfs = []
# Analysis of outputs
for rgn in ['NW', 'NE']:
    for output in ['p11_step3', 'p11_step6']:
        if output == 'p11_step3':
            dv = DVector.load(Path(
                fr"F:\Working\Land-Use\OUTPUTS_forecast_population\01_Intermediate Files\
                Output p11_age_g_{rgn}.hdf"))
        else:
            dv = DVector.load(Path(
                fr"F:\Working\Land-Use\OUTPUTS_forecast_population\01_Intermediate Files\
                Output p11_age_g_soc_{rgn}.hdf"))
        dv_translated = dv.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False
        )
        # aggregate segmentation
        dv2 = dv_translated.aggregate(['age_ntem'])
        dv3 = dv_translated.aggregate(['g'])
        dv4 = dv_translated.aggregate(['soc'])
        # stack columns and format
        dv2 = dv2.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv2['segmentation'] = 'age_ntem'
        dv2['filename'] = f'{output}'
        dv2['output code'] = f'{output}'
        dv2 = dv2[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]

        dv3 = dv3.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv3['segmentation'] = 'g'
        dv3['filename'] = f'{output}'
        dv3['output code'] = f'{output}'
        dv3 = dv3[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]

        dv4 = dv4.data.stack().reset_index().set_axis(['segment', 'region', 'value'], axis=1)
        dv4['segmentation'] = 'soc'
        dv4['filename'] = f'{output}'
        dv4['output code'] = f'{output}'
        dv4 = dv4[['filename', 'segmentation', 'output code', 'region', 'segment', 'value']]

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
final_output.to_csv(
    r'F:\Working\Land-Use\OUTPUTS_forecast_population\Analysis\outputs\population_forecast_output_summary.csv')
