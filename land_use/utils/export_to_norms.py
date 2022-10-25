# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:58:33 2022

@author: ElizabethForde

MSOA level pop + emp data to NoRMS zone conversion script
"""


import pandas as pd
import os
import bz2
import _pickle as cPickle


def merge_required_lookups(inputdf: pd.DataFrame,
                           metric: str):
    """
    """
    data = inputdf.copy()
    
    # MSOA sum
    msoa_total_sum = data[metric].sum()
    
    # merge to MSOA_NoRMS lookup
    if metric == 'employment':
        data = pd.merge(data, msoa_norms_lookup,
                        left_on=['msoa_zone_id'], right_on=['msoa11cd'],
                        how='left')
        
        # apply proportion
        data[metric] = data[metric] * data['msoa_to_norms']
        
        # then groupby norms zones
        data = data.groupby(['norms_zone_id', 
                             'e_cat',
                             'soc_cat']).agg({metric: 'sum'}).reset_index()    

    elif metric == 'people':
        data = pd.merge(data, msoa_norms_lookup,
                        left_on=['MSOA'], right_on=['msoa11cd'],
                        how='left')
        
        # apply proportion
        data[metric] = data[metric] * data['msoa_to_norms']
        
        # then groupby norms zones
        data = data.groupby(['norms_zone_id', 
                             'tfn_tt']).agg({metric: 'sum'}).reset_index()        

    # NoRMS zone sum: 
    norms_total_sum = data[metric].sum()

    return data, msoa_total_sum,  norms_total_sum


if __name__ == '__main__':

    scenarios = ['Nov 21 central', 'CAS High', 'CAS Low', 'CAS Regional Scenario']

    for scenario in scenarios:

        # --- filepaths:
        pop_emp_data_filepath = r'I:\NorMITs Land Use\future_land_use\iter4m\03 Outputs\%s' % scenario
        nelum_inputs_file_path = r'Y:\NoRMS\Accessibility Analysis\RMAP_Workstream\New Statics\nelum_inputs'
        temp_wd = r'I:\NorMITs Land Use\future_land_use\iter4m\03 Outputs\%s\NoRMS zone conversion' % scenario

        if not os.path.exists(temp_wd):
            os.mkdir(temp_wd)

        # --- MSOA to NoRMS zone lookup:
        msoa_norms_lookup = pd.read_csv(
            os.path.join(nelum_inputs_file_path, 'aggregations', 'MSOA_to_Norms1300.csv'))
        msoa_norms_lookup = msoa_norms_lookup[['msoa11cd', 'norms_zone_id', 'msoa_to_norms']]

        emp_zone_total_diff = {}
        pop_zone_total_diff = {}

        for file in os.listdir(pop_emp_data_filepath):

            if 'pop' in file and 'pbz2' in file:
                pop_data = cPickle.load(bz2.BZ2File(os.path.join(pop_emp_data_filepath, file), 'rb'))

                pop_output, MSOA_zone_sum, NoRMS_zone_sum = merge_required_lookups(inputdf=pop_data,
                                                                                   metric='people')
                # total diffs
                MSOA_NoRMS_difference = ((MSOA_zone_sum - NoRMS_zone_sum) / MSOA_zone_sum) * 100
                pop_zone_total_diff.update({file: MSOA_NoRMS_difference})

                # output files to csv
                file = file.split('.')[0]
                pop_output.to_csv(os.path.join(temp_wd, 'NoRMS_' + file + '.csv'), index=False)

            elif 'emp' in file and 'csv' in file:
                emp_data = pd.read_csv(os.path.join(pop_emp_data_filepath, file))

                emp_output, MSOA_zone_sum, NoRMS_zone_sum = merge_required_lookups(inputdf=emp_data,
                                                                                   metric='employment')

                # total diffs
                MSOA_NoRMS_difference = ((MSOA_zone_sum - NoRMS_zone_sum) / MSOA_zone_sum) * 100
                emp_zone_total_diff.update({file: MSOA_NoRMS_difference})

                # output files to csv
                emp_output.to_csv(os.path.join(temp_wd, 'NoRMS_' + file + '.csv'), index=False)

        employment_zone_total_diff = pd.DataFrame.from_dict(emp_zone_total_diff,
                                                            orient='index',
                                                            columns=['diff (%)']
                                                            ).to_csv(os.path.join(temp_wd, 'employment_zone_total_diff_NoRMS.csv'))

        population_zone_total_diff = pd.DataFrame.from_dict(pop_zone_total_diff,
                                                            orient='index',
                                                            columns=['diff (%)']
                                                            ).to_csv(os.path.join(temp_wd, 'population_zone_total_diff_NoRMS.csv'))