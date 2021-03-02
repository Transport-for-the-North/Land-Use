# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:01:59 2020

@author: mags15
Was built based on nts = pd.read_csv('Y:/NTS/tfn_ntem_build.csv') 
but can't find it now so rewritten to take 'tfn_unclassified_build.csv'

"""
import pandas as pd

_default_iter = 'iter4'
_default_home = 'E:/NorMITs_Export/'
_default_home_dir = (_default_home + _default_iter)
_import_folder = 'Y:/NorMITs Land Use/import/'
_import_file_drive = 'Y:/'
_default_zone_folder = ('I:/NorMITs Synthesiser/Zone Translation/Export/')
_default_area_types = ('Y:/NorMITs Land Use/area types/TfNAreaTypesLookup.csv')
_ward_to_msoa = _default_zone_folder + 'uk_ward_msoa_pop_weighted_lookup.csv'
_nts_path = 'Y:/NTS/import/tfn_unclassified_build.csv'

_hc_lookup = _import_folder+'Car availability/household_composition.csv'
_emp_lookup = _import_folder+'Car availability/emp_type.csv'
_adults_lookup = _import_folder +'Car availability/adults_lookup.csv'


def ntsimport (year = ['2017']):
    """
    imports car availability from nts extract 'tfn_unclassified_build'
    
    """
    ntsextract = pd.read_csv(_nts_path)
    ntsextract = ntsextract[ntsextract.SurveyYear.isin(year)]

    ntsextract = ntsextract[['HHoldOSWard_B01ID', 'HHoldNumAdults', 'NumCarVan_B02ID',
                             'W1', 'W2', 'EcoStat_B01ID']]
    ### NTS weightings ###
    # individual: weight = w1 *w2
    ntsextract['people'] = ntsextract['W1']*ntsextract['W2']
    ntsextract = ntsextract.rename(columns = {'HHoldOSWard_B01ID': 'uk_ward_zone_id'})
    # change household number of adults into household composition
    adults = pd.read_csv(_adults_lookup)
    ntsextract = ntsextract.merge(adults, on = ['HHoldNumAdults'])
    hc_lookup = pd.read_csv(_hc_lookup)

    hc_lookup = hc_lookup.rename(columns = {'household':'adults', 'cars':'NumCarVan_B02ID'})
    ntsextract = ntsextract.merge(hc_lookup, on = ['adults','NumCarVan_B02ID'])
    
    # change emp type
    emp_lookup = pd.read_csv(_emp_lookup)
    ntsextract = ntsextract.merge(emp_lookup, on = 'EcoStat_B01ID')
    
    cars = ntsextract.groupby(by=['uk_ward_zone_id', 'household_composition',  
                                'employment_type'], 
                                as_index = False).sum().drop(columns = {'W1', 'W2'})
    #null_values = [-8,-9]
    #cars = cars[~cars.NumCarVan_B02ID.isin(null_values)]
    #cars = cars[~cars.EcoStat_B01ID.isin(null_values)]
    cars['people'].sum()
    wardToMSOA = pd.read_csv(_ward_to_msoa).drop(columns={'overlap_var', 
                          'uk_ward_var', 'msoa_var', 'overlap_type', 'overlap_msoa_split_factor'})

    ####### lots of zeros for MSOAs totals and other segments so not able to use it ###############
    cars_msoa = cars.merge(wardToMSOA, on = 'uk_ward_zone_id')
    cars_msoa['population'] = cars_msoa['people'] * cars_msoa['overlap_uk_ward_split_factor']
    cars_msoa['population'].sum()
    cars_msoa = cars_msoa.groupby(by=['msoa_zone_id', 'household_composition',
                                       'employment_type'],
                                        as_index = False).sum().drop(columns=
                                                             {'overlap_uk_ward_split_factor', 
                                                              'people'})
    cars_msoa['population'].sum()
    
    # read in the new areatypes
    areatypes = pd.read_csv(_default_area_types).rename(columns={'msoa_area_code':'msoa_zone_id', 
                           'tfn_area_type_id':'area_type'})
    cars_msoa = cars_msoa.merge(areatypes, on = 'msoa_zone_id') 
    
    # per area type - MSOA had too small sample sizes
    cars_at = cars_msoa.groupby(by=['area_type', 'household_composition', 
                                'employment_type'], 
                                as_index = False).sum()
    cars_at['totals'] = cars_at.groupby(['area_type','employment_type'])['population'].transform('sum')
    
    cars_at['splits'] = cars_at['population']/cars_at['totals']

    cars_at['splits'].sum()
    # check the number of survey results per area type
    cars_n = cars_at.groupby(by=['area_type'], as_index = False).sum().reindex(columns={'area_type', 'population'})
    cars_at = cars_at.reindex(columns={'area_type', 'household_composition', 'employment_type', 'splits'})
    cars_at.to_csv(_default_home_dir+'/nts_splits.csv', index = False)
    return(cars_at)
    
    del(cars_msoa, cars, ntsextract)