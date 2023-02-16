# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 2021 - Thu Oct 28 2021

@author: adamtebbs
Version number:

Written using: Python 3.7.1

Module versions used for writing:
    pandas v0.25.3
    numpy v1.17.3

census2011_population_furness:
    - imports NTEM mid-year population estimates and 2011 Census micro data
    - applies 2011 census segmentation
    - and ns-sec and soc segmentation
    - runs the IPFN process
    - does not process communal establishments
    - outputs f factor to be used in 2018
## TODO: Needs checking. Also need to ensure that the final f factors are actually produced at some point!
## TODO: Ensure this script gets called properly! NB - DO NOT ACTUALLY RUN IT! It would take weeks to finish!
# Actually looks like 1 full day if the districts are consistent

# CS
# It has to run unfortunately, all of these tools need to work 3rd party
# TODO: Optimise
# TODO: Can this inherit write strings from the main objects - they pop out everywhere
"""

import pandas as pd
import numpy as np
import os
import itertools
import datetime
import pyodbc
from functools import reduce
from os.path import join as opj
# from ipfn import ipfn
from caf.toolkit import iterative_proportional_fitting as ipfn
from caf.toolkit import concurrency
import logging

# TODO: These should come from a constants paths that a 3rd party user can parse
# That includes making the pathing relative

# Data input paths
_census_micro_path = 'I:/NorMITs Land Use/import/2011 Census Microdata'
_QS_census_queries_path = 'I:/NorMITs Land Use/import/Nomis Census 2011 Head & Household'
_lookup_tables_path = 'I:/NorMITs Land Use/import/2011 Census Micro lookups'
_NTEM_input_path = r'I:\NorMITs Land Use\import\CTripEnd'
_lookup_paths = {
    'age': opj(_lookup_tables_path, 'ageh.csv'),
    'gender': opj(_lookup_tables_path, 'sex.csv'),
    'alt_hh_comp': opj(_lookup_tables_path, 'ahchuk11.csv'),
    'cars': opj(_lookup_tables_path, 'carsnoc.csv'),
    'hh_comp': opj(_lookup_tables_path, 'h.csv'),
    'econ_activity': opj(_lookup_tables_path, 'ecopuk11.csv'),
    'hours': opj(_lookup_tables_path, 'hours.csv'),
    'nssec': opj(_lookup_tables_path, 'nsshuk11.csv'),
    'soc': opj(_lookup_tables_path, 'occg.csv'),
    'type_accom': opj(_lookup_tables_path, 'Typaccom.csv'),
    'geography': opj(_lookup_tables_path, 'geography.csv'),
    'ntem_tt': opj(_NTEM_input_path, 'Pop_Segmentations.csv')}


# lookup_geography_filename = 'lookup_geography_z2d2r.csv'
# lookup_folder = r'I:\NorMITs Land Use\import\2011 Census Furness\04 Post processing\Lookups'


# Define model name and output folder
# Note that output folder is IPFN input folder
ModelName = 'NorMITs'
Output_Folder = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs'

# Segmentation keys
aghe = ['a', 'g', 'h', 'e']
tns = ['t', 'n', 's']
zdr = ['z', 'd', 'r']

def _load_lookup_data(path_dict):
    lookups = dict()

    # --- Conversions/mappings ---
    # Age -> (a)
    age = pd.read_csv(path_dict["age"])[["Age", "NorMITs_Segment Band Value"]]
    age = age.set_index("Age").rename(columns={"NorMITs_Segment Band Value": "a"})
    lookups["age"] = age.astype(int)
    # Sex -> (g)
    gender = pd.read_csv(path_dict["gender"])[["Sex", "NorMITs_Segment Band Value"]]
    gender = gender.set_index("Sex").rename(columns={"NorMITs_Segment Band Value": "g"})
    lookups["gender"] = gender.astype(int)
    # Household composition -> (h)
    hh_comp = pd.read_csv(path_dict["hh_comp"])[["Household Composition Key", "Household_composition_code"]]
    hh_comp = hh_comp.set_index("Household Composition Key").rename(columns={"Household_composition_code": "h"})
    lookups["hh_comp"] = hh_comp.astype(int)
    # Economic activity -> (e)
    econ_activity = pd.read_csv(path_dict["econ_activity"])[["Employment type code", "NorMITs_Segment Band Value"]]
    econ_activity = econ_activity.set_index("Employment type code").rename(columns={"NorMITs_Segment Band Value": "e"})
    lookups["econ_activity"] = econ_activity
    # NS-SEC -> (n)
    nssec = pd.read_csv(path_dict["nssec"])[["HRP NSSEC", "NorMITs_Segment Band Value"]]
    nssec = nssec.set_index("HRP NSSEC").rename(columns={"NorMITs_Segment Band Value": "n"})
    lookups["nssec"] = nssec.astype(int)
    # SOC -> (s)
    soc = pd.read_csv(path_dict["soc"])[["SOC", "NorMITs_Segment Band Value"]]
    soc = soc.set_index("SOC").rename(columns={"NorMITs_Segment Band Value": "s"})
    lookups["soc"] = soc.astype(int)
    # Hours worked -> (_FT-PT)
    hours = pd.read_csv(path_dict["hours"])[["Hours worked ", "NorMITs_Segment Band Value"]]
    hours = hours.set_index("Hours worked ").rename(columns={"NorMITs_Segment Band Value": "_FT-PT"})
    lookups["ft_pt"] = hours.astype(int)
    # Alternate household compsotion -> (_Adults)
    adults = pd.read_csv(path_dict["alt_hh_comp"])[["Household size", "NorMITs_Segment Band Value"]]
    adults = adults.set_index("Household size").rename(columns={"NorMITs_Segment Band Value": "_Adults"})
    lookups["adults"] = adults.astype(int)
    # Cars in the household -> (_Cars)
    cars = pd.read_csv(path_dict["cars"])[["Household car", "NorMITs_Segment Band Value"]]
    cars = cars.set_index("Household car").rename(columns={"NorMITs_Segment Band Value": "_Cars"})
    lookups["cars"] = cars.astype(int)
    # NTEM traveller type -> (a,g,h,e)
    ntem_tt = pd.read_csv(path_dict["ntem_tt"])[["NTEM_Traveller_Type",
                                                 "Age_code",
                                                 "Gender_code",
                                                 "Household_composition_code",
                                                 "Employment_type_code"]]
    ntem_tt = ntem_tt.set_index("NTEM_Traveller_Type").rename(columns={"Age_code": "a",
                                                                       "Gender_code": "g",
                                                                       "Household_composition_code": "h",
                                                                       "Employment_type_code": "e"})
    lookups["ntem_tt"] = ntem_tt.astype(int)

    # --- Additional data ---
    geography = pd.read_csv(path_dict["geography"])[['NorMITs Zone', "Grouped LA", 'NorMITs Region', "MSOA"]]
    geography = geography.rename(columns={'NorMITs Zone': 'z', 'Grouped LA': 'd', 'NorMITs Region': 'r'})
    geography = _generate_scottish_geography(geography_lookup=geography)
    lookups["geography"] = geography

    return lookups


def _generate_scottish_geography(geography_lookup):

    # Sort out district lookups and apply 'districts' to Scotland
    # by grouping zones numerically to the EW average district size.

    geography = geography_lookup.copy()

    geography_EW = geography.loc[geography["MSOA"].str[0].isin(["E", "W"])].copy()
    geography_EW = geography_EW.reset_index(drop=True)
    geography_EW['d'] = geography_EW['d'].astype(int)

    EW_zones_per_district = geography_EW.groupby(['d'], as_index=False)['z'].nunique()
    avg_EW_zones_per_district = round(EW_zones_per_district['z'].mean())
    max_EW_zone = EW_zones_per_district['d'].max()

    geography_S = geography.loc[geography["MSOA"].str[0] == "S"].copy()
    geography_S = geography_S.reset_index(drop=True)
    geography_S['scottish_z'] = geography_S.index
    geography_S['d'] = (geography_S['scottish_z'] // avg_EW_zones_per_district) + 1
    geography_S['d'] = geography_S['d'] + max_EW_zone
    geography_S['r'] = "Scotland"

    geography_GB = pd.concat([geography_EW, geography_S], axis=0, ignore_index=True)[zdr+['MSOA']]
    geography_GB[['z', 'd']] = geography_GB[['z', 'd']].astype(int)
    return geography_GB


def _load_population_data(census_microdata_path, QS401_path, QS606_path, QS609_path, NTEM_population_path):
    census_microdata = pd.read_csv(census_microdata_path)
    census_microdata = census_microdata.rename(columns={
        'la_group': 'd', 'typaccom': 't',
        'ageh': 'Age', 'sex': 'Sex', 'nsshuk11': 'HRP NSSEC',
        'ahchuk11': 'Household size', 'carsnoc': 'Household car',
        'ecopuk11': 'Employment type code', 'hours': 'Hours worked', 'occg': 'SOC'})
    census_microdata = census_microdata[['caseno', 'd', 'residence_type', 't',
                                         'Age', 'Sex', 'HRP NSSEC',
                                         'Household size', 'Household car',
                                         'Employment type code', 'Hours worked', 'SOC']]

    QS401 = pd.read_csv(QS401_path, skiprows=7)
    QS401 = QS401.rename(columns={'All categories: Accommodation type': 'Census_Population', "mnemonic": "MSOA"})

    QS606 = pd.read_csv(QS606_path, skiprows=7)
    QS606 = QS606.rename(columns={'All categories: Occupation': 'Census_Population', 'mnemonic': 'MSOA'})

    QS609 = pd.read_csv(QS609_path, skiprows=6)
    QS609 = QS609.rename(columns={'All categories: NS-SeC': 'Census_Population', "mnemonic": "MSOA"})

    # Trim the footers off the tables (they are always in the 2nd column, so dropna on Area (1st column))
    QS401 = QS401.dropna(subset=['Area'])
    QS606 = QS606.dropna(subset=['Area'])
    QS609 = QS609.dropna(subset=['Area'])

    NTEM_population = pd.read_csv(NTEM_population_path)
    NTEM_population = NTEM_population.rename(columns={'2011': 'C_NTEM', 'tt': 'NTEM_tt'})
    return census_microdata, QS401, QS606, QS609, NTEM_population


def _segment_and_tally_census_microdata(census_microdata, segment_lookup):
    """
    Convert census segmentation to NorMITs segmentation. Count each unique segmentation combination.

    :param census_microdata:
        Dataframe of census microdata, as generated by _load_population_data
    :param segment_lookup:
        Dictionary of lookup dataframes mapping between NTEM and NorMITs segmentation.
        Each dataframe generally contains a NorMITs_Segment Band Value column.
        Keys ['nssec','age','sex', 'hours','econ_activity','soc','alt_hh_comp','cars','hh_comp']
    :return:
        Number of cases in microdata for those living in a household (not communal) by NorMITs segmentation.
        Columns ['d','a','g','h','e','t','n','s']
        District, age, gender, household composition, employment type, accomodation type, NSSEC, SOC
    """

    hh_microdata = census_microdata.loc[census_microdata["residence_type"] == 2].copy()
    hh_microdata = hh_microdata.dropna(subset=["Household size", "HRP NSSEC"], how="any")

    # Age
    hh_microdata = hh_microdata.join(segment_lookup["age"], on="Age", validate="m:1")
    hh_microdata = hh_microdata.drop(columns=["Age"])
    # Gender
    hh_microdata = hh_microdata.join(segment_lookup["gender"], on="Sex", validate="m:1")
    hh_microdata = hh_microdata.drop(columns=["Sex"])
    hh_microdata.loc[hh_microdata["a"] == 1, "g"] = 1
    # Household Composition
    hh_microdata = hh_microdata.join(segment_lookup["adults"], on="Household size", validate="m:1")
    hh_microdata = hh_microdata.join(segment_lookup["cars"], on="Household car", validate="m:1")
    hh_microdata['Household Composition Key'] = hh_microdata[['_Adults', '_Cars']].astype(str).agg('_'.join, axis=1)
    hh_microdata = hh_microdata.join(segment_lookup["hh_comp"], on="Household Composition Key", validate="m:1")
    hh_microdata = hh_microdata.drop(columns=["Household size", "_Adults",
                                              "Household car", "_Cars",
                                              "Household Composition Key"])
    # Employment type
    # Consider age (children and retirees cannot work), if students (ecopuk11 type 8) are fte or pte via hours
    hh_microdata = hh_microdata.join(segment_lookup["ft_pt"], on="Hours worked", validate="m:1")
    hh_microdata["_FT-PT"] = hh_microdata["_FT-PT"].fillna(2)
    hh_microdata = hh_microdata.join(segment_lookup["econ_activity"], on="Employment type code", validate="m:1")
    hh_microdata.loc[hh_microdata["Employment type code"] == 8, 'e'] = hh_microdata['_FT-PT']   # TODO: Is this safe?
    hh_microdata.loc[hh_microdata["a"] != 2, 'e'] = 5
    hh_microdata = hh_microdata.drop(columns=["Hours worked", "_FT-PT", "Employment type code"])
    hh_microdata = hh_microdata.dropna(subset=['e'])
    # NSSEC
    hh_microdata = hh_microdata.join(segment_lookup["nssec"], on="HRP NSSEC", validate="m:1")
    hh_microdata = hh_microdata.drop(columns=["HRP NSSEC"])
    # SOC
    hh_microdata = hh_microdata.join(segment_lookup["soc"], on="SOC", validate="m:1")
    hh_microdata.loc[hh_microdata['e'].astype(int) > 2, 's'] = 4
    hh_microdata = hh_microdata.drop(columns=["SOC"])

    # Type and column name formatting
    hh_microdata[['e']+tns] = hh_microdata[['e']+tns].astype(int)
    hh_microdata_count = hh_microdata.groupby(['d']+aghe+tns)['caseno'].nunique().reset_index()
    hh_microdata_count = hh_microdata_count.rename(columns={'caseno': "C_daghetns"})
    return hh_microdata_count


def _generate_all_valid_population_segments(microdata_count, aghe_segment_count, geography_lookup):
    """
    Find all valid (a,g,h,e) combinations, and all valid (a,g,h,e,t,n,s) noting the worker/non_worker restrictions.

    :param microdata_count:
        Number of cases in microdata for each (d,a,g,h,e,t,n,s) combination,
        as generated by _segment_and_tally_census_microdata
    :param aghe_segment_count:
        Expected number of (a,g,h,e) combinations
    :param geography_lookup:
        Dataframe with (z)one/(d)istrict/(r)egion correspondence.
    :return:
        Dataframes with all valid segment combinations of:
            (a,g,h,e) applied to all model districts - Columns ['d','a','g','h','e']
            (a,g,h,e,t,n,s) - Columns ['a','g','h','e','t','n','s']
    """

    microdata_segments = microdata_count[aghe+tns].copy()
    microdata_segments = microdata_segments.drop_duplicates()

    # ---- All population (a, g, h, e) segmentations ----
    # All valid population (a, g, h, e) segmentations
    aghe_segments = microdata_segments[aghe].drop_duplicates()
    # Check a,g,h,e combinations match to all NTEM traveller types
    if len(aghe_segments) == aghe_segment_count:
        print('No globally missing NTEM_tt')
    else:
        print('INCORRECT GLOBAL NTEM_tt TOTAL!')
        print('Expected', aghe_segment_count)
        print('Got', len(aghe_segments))

    # All valid population (a,g,h,e) segmentations, for all districts
    model_districts = geography_lookup.loc[geography_lookup['r'] != "Scotland", 'd']
    daghe_segments = itertools.product(
        model_districts.unique(),
        aghe_segments.itertuples(index=False))
    daghe_segments = pd.DataFrame(daghe_segments, columns=["d", "aghe"])
    daghe_segments[aghe] = pd.DataFrame(daghe_segments["aghe"].to_list())
    daghe_segments = daghe_segments.drop(columns=["aghe"])

    # ---- All population (a, g, h, e, t, n, s) segmentations ----
    # All valid worker (a, g, h, e, t, n, s) segmentations
    worker = microdata_segments.loc[(microdata_segments['e'] <= 2) & (microdata_segments['s'] < 4)]
    worker_segments = itertools.product(
        worker[aghe].drop_duplicates().itertuples(index=False),
        worker["t"].unique(), worker["n"].unique(), worker["s"].unique())
    worker_segments = pd.DataFrame(worker_segments, columns=["aghe"]+tns)

    # All valid non-worker (a, g, h, e, t, n, s) segmentations
    non_worker = microdata_segments.loc[(microdata_segments['e'] > 2) & (microdata_segments['s'] == 4)]
    non_worker_segments = itertools.product(
        non_worker[aghe].drop_duplicates().itertuples(index=False),
        non_worker["t"].unique(), non_worker["n"].unique(), non_worker["s"].unique())
    non_worker_segments = pd.DataFrame(non_worker_segments, columns=["aghe"]+tns)

    aghetns_segments = pd.concat([worker_segments, non_worker_segments], axis=0, ignore_index=True)
    aghetns_segments[aghe] = pd.DataFrame(aghetns_segments["aghe"].to_list())
    aghetns_segments = aghetns_segments.drop(columns=["aghe"])
    return daghe_segments, aghetns_segments


def _estimate_f_tns_daghe(microdata_count, aghe_segment_count, daghe_segments):
    """
    Calculate F(t,n,s|d,a,g,h,e), the proportion of (d,a,g,h,e) count which are (t,n,s),
    and F(t,n,s|a,g,h,e), the England and Wales proportion of (a,g,h,e) which are (t,n,s).

    :param microdata_count:
        Number of cases in microdata for each (d,a,g,h,e,t,n,s) combination,
        as generated by _segment_and_tally_census_microdata.
    :param aghe_segment_count:
        Expected number of (a,g,h,e) combinations.
    :param daghe_segments:
        Dataframes with all valid segment combinations of (d,a,g,h,e).
    :return:
        Dataframes:
         infilled F(t,n,s|d,a,g,h,e) - Columns ['d','a','g','h','e','F(t,n,s|d,a,g,h,e)']
         F(t,n,s|d,a,g,h,e) - Columns ['d','a','g','h','e','F(t,n,s|d,a,g,h,e)']
         F(t,n,s|a,g,h,e) - Columns ['a','g','h','e','F(t,n,s|d,a,g,h,e)']
    """
    # F(t,n,s|d,a,g,h,e) -> Fraction of the count (d,a,g,h,e) which are (t,n,s)
    f_tns_daghe = microdata_count.copy()
    f_tns_daghe["C_daghe"] = microdata_count.groupby(["d"]+aghe)["C_daghetns"].transform("sum")
    f_tns_daghe['F(t,n,s|d,a,g,h,e)'] = f_tns_daghe['C_daghetns'] / f_tns_daghe['C_daghe']
    f_tns_daghe = f_tns_daghe.drop(columns=['C_daghetns', 'C_daghe'])

    # F(t,n,s|d=E&W,a,g,h,e)=F(t,n,s|a,g,h,e) -> Fraction of the count in England and Wales (a,g,h,e) which are (t,n,s)
    f_tns_aghe = microdata_count.copy().rename(columns={"C_daghetns": "C_aghetns"})
    f_tns_aghe = f_tns_aghe.groupby(aghe+tns, as_index=False)["C_aghetns"].sum()
    f_tns_aghe["C_aghe"] = f_tns_aghe.groupby(aghe)["C_aghetns"].transform("sum")
    f_tns_aghe['F(t,n,s|a,g,h,e)'] = f_tns_aghe['C_aghetns'] / f_tns_aghe['C_aghe']
    f_tns_aghe = f_tns_aghe.drop(columns=['C_aghetns', 'C_aghe'])

    # Find and replace missing zonal Census micro NTEM traveller types with EW averages
    missing_f_tns_daghe = daghe_segments.merge(f_tns_daghe, how="left", on=["d"]+aghe, validate="1:m")
    missing_f_tns_daghe = missing_f_tns_daghe.loc[missing_f_tns_daghe.isnull().any(axis=1), ["d"]+aghe]
    missing_f_tns_daghe = missing_f_tns_daghe.merge(f_tns_aghe, how="left", on=aghe, validate="m:m")
    missing_f_tns_daghe = missing_f_tns_daghe.rename(columns={"F(t,n,s|a,g,h,e)": "F(t,n,s|d,a,g,h,e)"})
    infilled_f_tns_daghe = pd.concat([f_tns_daghe, missing_f_tns_daghe], axis=0, ignore_index=True)

    EW_f_by_d = round(infilled_f_tns_daghe['F(t,n,s|d,a,g,h,e)'].sum()) / infilled_f_tns_daghe['d'].nunique()
    if EW_f_by_d == aghe_segment_count:
        print('f combinations appear valid')
    else:
        print('ISSUE WITH f PROCESSING!')

    return infilled_f_tns_daghe, f_tns_daghe, f_tns_aghe


def _segment_and_scale_ntem_population(NTEM_population, f_tns_daghe, ntem_tt_lookup, geography_lookup):
    """
    Use F(t,n,s|d,a,g,h,e) to split NTEM population into full segmentation.

    :param NTEM_population:
        NTEM population data
    :param f_tns_daghe:
        F(t,n,s|d,a,g,h,e), the proportion of (d,a,g,h,e) count which are (t,n,s)
    :param ntem_tt_lookup:
        Mapping from NTEM traveller type to (a,g,h,e)
    :return:

    """

    print(NTEM_population)
    print('\n')
    print(ntem_tt_lookup)

    NTEM_population = NTEM_population.copy()
    NTEM_population = NTEM_population.join(ntem_tt_lookup, on="NTEM_tt", validate="m:1")

    cols_chosen = ['z', 'A', 'NTEM_tt', 'a', 'g', 'h', 'e', 'C_NTEM']
    NTEM_population = NTEM_population[cols_chosen]
    # NTEM_population = ntem_pop_interpolation(census_and_by_lu_obj)
    # NTEM_population = NTEM_population[cols_chosen]
    print('Total 2011 ntem household population is : ', NTEM_population["C_NTEM"].sum())

    NTEM_pop_actual = NTEM_population.copy()

    # Drop the Scottish districts and apply f to England and Wales
    NTEM_pop_actual = NTEM_pop_actual.merge(geography_lookup, how='left', on='z', validate='m:1')
    NTEM_pop_EW = NTEM_pop_actual.loc[NTEM_pop_actual["r"] != "Scotland"]
    test_tot_EW = NTEM_pop_EW['C_NTEM'].sum()
    NTEM_pop_EW = NTEM_pop_EW.merge(f_tns_daghe, how="left", on=["d"]+aghe, validate='m:m')

    # Filter to obtain just North East/North West.
    # Recalculate f by Area type (A) for these regions.
    NTEM_pop_N = NTEM_pop_EW.copy()
    NTEM_pop_N = NTEM_pop_N.loc[NTEM_pop_N['r'].isin(['North East', 'North West'])]
    NTEM_pop_N = NTEM_pop_N.groupby(['A']+aghe+tns, as_index=False)['C_NTEM'].sum()
    NTEM_pop_N['F(t,n,s|A,a,g,h,e)'] = NTEM_pop_N["C_NTEM"] / NTEM_pop_N.groupby(['A']+aghe)['C_NTEM'].transform("sum")
    NTEM_pop_N = NTEM_pop_N[["A"]+aghe+tns+["F(t,n,s|A,a,g,h,e)"]]

    NTEM_pop_S = NTEM_pop_actual.copy()
    NTEM_pop_S = NTEM_pop_S.loc[NTEM_pop_S["r"] == "Scotland"]
    test_tot_S = NTEM_pop_S['C_NTEM'].sum()
    NTEM_pop_S = NTEM_pop_S.merge(NTEM_pop_N, how="left", on=["A"]+aghe, validate='m:m')

    NTEM_pop_S = NTEM_pop_S.rename(columns={"F(t,n,s|A,a,g,h,e)": "F(t,n,s|z,a,g,h,e)"})  # As A=A(z)
    NTEM_pop_EW = NTEM_pop_EW.rename(columns={"F(t,n,s|d,a,g,h,e)": "F(t,n,s|z,a,g,h,e)"})  # As d=d(z)

    NTEM_pop_scaled = pd.concat([NTEM_pop_EW, NTEM_pop_S], axis=0, ignore_index=True)
    NTEM_pop_scaled['C_zaghetns'] = NTEM_pop_scaled['F(t,n,s|z,a,g,h,e)'] * NTEM_pop_scaled['C_NTEM']

    # Print some totals out to check...
    print('Actual EW tot:' + str(test_tot_EW))  # Matches scaled population to 1/1000 of a fingernail
    print('Scaled EW tot:' + str((NTEM_pop_EW['C_NTEM']*NTEM_pop_EW['F(t,n,s|z,a,g,h,e)']).sum()))
    print('Actual S tot: ' + str(test_tot_S))
    print('Scaled S tot: ' + str((NTEM_pop_S['C_NTEM']*NTEM_pop_S['F(t,n,s|z,a,g,h,e)']).sum()))
    print('Actual GB tot:' + str(test_tot_S + test_tot_EW))
    print('Scaled GB tot:' + str((NTEM_pop_scaled['C_zaghetns']).sum()))
    return NTEM_pop_actual, NTEM_pop_scaled


def _generate_population_seeds(NTEM_population, aghetns_segments):

    # The following block takes ~ 3 minutes.
    all_z_aghetns = itertools.product(
        NTEM_population[zdr].drop_duplicates().itertuples(index=False),
        aghetns_segments[aghe+tns].drop_duplicates().itertuples(index=False))
    all_z_aghetns = pd.DataFrame(all_z_aghetns, columns=["zdr", "aghetns"])
    all_z_aghetns[zdr] = pd.DataFrame(all_z_aghetns["zdr"].to_list())
    all_z_aghetns[aghe+tns] = pd.DataFrame(all_z_aghetns["aghetns"].to_list())
    all_z_aghetns = all_z_aghetns.drop(columns=["zdr", "aghetns"])

    NTEM_pop_for_seeds = NTEM_population.groupby(zdr+aghe+tns+["NTEM_tt"], as_index=False)["C_zaghetns"].sum()
    NTEM_pop_for_seeds = NTEM_pop_for_seeds.merge(all_z_aghetns, how="right", on=zdr+aghe+tns, validate="1:1")

    NTEM_pop_for_seeds[["z", "d"]+aghe+tns] = NTEM_pop_for_seeds[["z", "d"]+aghe+tns].astype(int)
    NTEM_pop_for_seeds['C_zaghetns'] = NTEM_pop_for_seeds['C_zaghetns'].fillna(0)

    NTEM_pop_for_seeds['NTEM_tt'] = NTEM_pop_for_seeds['NTEM_tt'].str[-3:].astype(float)
    NTEM_pop_for_seeds['NTEM_tt'] = NTEM_pop_for_seeds.groupby(aghe)['NTEM_tt'].transform("mean")
    NTEM_pop_for_seeds['NTEM_tt'] = NTEM_pop_for_seeds['NTEM_tt'].astype(int)


    seed_r_NW = NTEM_pop_for_seeds.loc[NTEM_pop_for_seeds['r'] == 'North West']
    seed_r_NW = seed_r_NW.reset_index()[['z']+aghe+tns+['C_zaghetns']]
    seed_d_184 = NTEM_pop_for_seeds.loc[NTEM_pop_for_seeds['d'] == 184]
    seed_d_184 = seed_d_184.reset_index()[['z']+aghe+tns+['C_zaghetns']]

    seed_population = NTEM_pop_for_seeds[zdr+aghe+tns+['C_zaghetns']]
    seed_population = seed_population.rename(columns={"C_zaghetns": "population"})
    return seed_population


def _segment_qs(census_QS, NTEM_population, segment_letter, column_to_segment_mapping, segment_to_id_mapping):

    column_to_segment_mapping = {k: f"__{v}__" for k, v in column_to_segment_mapping.items()}
    segment_to_id_mapping = {f"__{k}__": v for k, v in segment_to_id_mapping.items()}

    NTEM_population = NTEM_population.copy()
    NTEM_population = NTEM_population.groupby(['z', 'd', 'r', 'MSOA'])['C_NTEM'].sum().reset_index()

    census_QS = census_QS.copy()
    census_QS = census_QS.rename(columns=column_to_segment_mapping)
    census_QS = census_QS.groupby(level=0, axis=1).sum()  # Sum columns with same name

    census_QS = census_QS.merge(NTEM_population, how="left", on='MSOA', validate="1:1")
    census_QS['Scaler'] = census_QS['C_NTEM'] / census_QS['Census_Population']
    census_QS = census_QS.melt(id_vars=['z', 'd', 'r', 'Scaler'],
                               value_vars=segment_to_id_mapping.keys(),
                               var_name="Segment", value_name="Persons")
    census_QS["Persons"] = census_QS["Persons"] * census_QS['Scaler']
    census_QS[segment_letter] = census_QS["Segment"].replace(segment_to_id_mapping)
    census_QS = census_QS.sort_values(by=['z', segment_letter]).reset_index()
    census_QS = census_QS[['z', 'd', 'r', segment_letter, 'Persons']]
    return census_QS


def _write_ipfn_inputs(seed_population, QS401, QS606, QS609, district_count):
    # os.chdir(Output_Folder)

    seed_file_path = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\01 Seed Files'
    NTEM_file_path = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\02 Ctrl1 NTEM Control Files'
    SOC_file_path = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\03 SOC Control Files'
    NSSEC_file_path = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\04 NSSEC Control Files'
    DT_file_path = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\05 DT Control Files'
    check_output_file_path = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs'

    Ctrl1_NTEM = seed_population.groupby(zdr+aghe, as_index=False)["population"].sum()

    print('Creating seed and control files')
    print('Printing every 10th district as they are written out:')

    # TODO: Slow process, needs mp wrapper
    for district in range(1, district_count+1):
        # Lookups and processing
        #     QS Control files
        QS401_d = QS401.loc[QS401['d'] == district]
        QS606_d = QS606.loc[QS606['d'] == district]
        QS609_d = QS609.loc[QS609['d'] == district]
        #     Main seed file
        seed_d = seed_population.loc[seed_population['d'] == district]
        seed_d = seed_d[["z"]+aghe+tns+["population"]]
        #     Ctrl1 control file
        Ctrl1_NTEM_d = Ctrl1_NTEM.loc[Ctrl1_NTEM['d'] == district]
        Ctrl1_NTEM_d = Ctrl1_NTEM_d[["z"]+aghe+["population"]]

        # TODO: Optimise writes, this is why it is so slow
        QS606_d.to_csv(opj(SOC_file_path, f"2011_{ModelName}_Ctrl_SOC_d_{district}_v0.1.csv"), index=False)
        QS609_d.to_csv(opj(NSSEC_file_path, f"2011_{ModelName}_Ctrl_NSSEC_d_{district}_v0.1.csv"), index=False)
        QS401_d.to_csv(opj(DT_file_path, f"2011_{ModelName}_Ctrl_DT_d_{district}_v0.1.csv"), index=False)
        seed_d.to_csv(opj(seed_file_path, f"2011_{ModelName}_seed_d_{district}_v0.1.csv"), index=False)
        Ctrl1_NTEM_d.to_csv(opj(NTEM_file_path, f"2011_{ModelName}_Ctrl1_NTEM_d_{district}_v0.1.csv"), index=False)

        # Print out every tenth row to check on progress
        if district / 10 == district // 10:
            print(district)
    print('All district level seed and control files have been printed out as csvs')
    print('Now working on checks...')

    # Check control file totals
    QS401_check_tot = QS401.groupby(['z'])['Persons'].sum().reset_index(name="QS401_pop")
    QS606_check_tot = QS606.groupby(['z'])['Persons'].sum().reset_index(name="QS606_pop")
    QS609_check_tot = QS609.groupby(['z'])['Persons'].sum().reset_index(name="QS609_pop")
    seed_check_tot = seed_population.groupby(['z'])['population'].sum().reset_index(name="Seed_pop")
    Ctrl1_check_tot = Ctrl1_NTEM.groupby(['z'])['population'].sum().reset_index(name="Ctrl1_pop")

    check_totals = reduce(lambda left, right: pd.merge(left, right, how="left", on="z", validate="1:1"),
                          [QS401_check_tot, QS606_check_tot, QS609_check_tot, seed_check_tot, Ctrl1_check_tot])

    check_totals.to_csv(opj(check_output_file_path, 'check_seed+control_totals.csv'), index=False)
    print('Checks completed and dumped to csv')
    return None


def create_ipfn_inputs(census_and_by_lu_obj):

    lookup_dict = _load_lookup_data(path_dict=_lookup_paths)

    census_microdata, QS401, QS606, QS609, NTEM_population = _load_population_data(
        census_microdata_path=opj(_census_micro_path, "recodev12.csv"),
        QS401_path=opj(_QS_census_queries_path, "210817_QS401UK -Dwelling type - Persons_MSOA.csv"),
        QS606_path=opj(_QS_census_queries_path, "210817_QS606UK - Occupation- ER_MSOA.csv"),
        QS609_path=opj(_QS_census_queries_path, "210817_QS609UK - NS-SeC of HRP- Persons_MSOA.csv"),
        NTEM_population_path=opj(_NTEM_input_path, 'All_year', 'ntem_gb_z_areatype_ntem_tt_2011_pop.csv'))

    hh_micro_count = _segment_and_tally_census_microdata(
        census_microdata=census_microdata,
        segment_lookup=lookup_dict)

    daghe_segments, aghetns_segments = _generate_all_valid_population_segments(
        microdata_count=hh_micro_count,
        aghe_segment_count=len(lookup_dict["ntem_tt"]),
        geography_lookup=lookup_dict["geography"])

    infilled_f_tns_daghe, f_tns_daghe, f_tns_aghe = _estimate_f_tns_daghe(
        microdata_count=hh_micro_count,
        aghe_segment_count=len(lookup_dict["ntem_tt"]),
        daghe_segments=daghe_segments)

    actual_NTEM_population, scaled_NTEM_population = _segment_and_scale_ntem_population(
        NTEM_population=NTEM_population,
        f_tns_daghe=infilled_f_tns_daghe,
        ntem_tt_lookup=lookup_dict["ntem_tt"],
        geography_lookup=lookup_dict["geography"])

    seed_population = _generate_population_seeds(
        NTEM_population=scaled_NTEM_population,
        aghetns_segments=aghetns_segments)

    _QS401 = _segment_qs(
        census_QS=QS401,
        NTEM_population=actual_NTEM_population,
        segment_letter='t',
        column_to_segment_mapping={
            "Unshared dwelling: Whole house or bungalow: Detached": "Detached",
            "Unshared dwelling: Whole house or bungalow: Semi-detached": "Semi-detached",
            "Unshared dwelling: Whole house or bungalow: Terraced (including end-terrace)": "Terraced",
            "Unshared dwelling: Flat, maisonette or apartment: Total": "Flat",
            "Unshared dwelling: Caravan or other mobile or temporary structure": "Flat",
            "Shared dwelling": "Flat"},
        segment_to_id_mapping={
            "Detached": 1,
            "Semi-detached": 2,
            "Terraced": 3,
            "Flat": 4})

    _QS606_work = _segment_qs(
        census_QS=QS606,
        NTEM_population=actual_NTEM_population.loc[actual_NTEM_population["e"] < 3],
        segment_letter='s',
        column_to_segment_mapping={
            '1. Managers, directors and senior officials': 'higher',
            '2. Professional occupations': 'higher',
            '3. Associate professional and technical occupations': 'higher',
            '4. Administrative and secretarial occupations': 'medium',
            '5. Skilled trades occupations': 'medium',
            '6. Caring, leisure and other service occupations': 'medium',
            '7. Sales and customer service occupations': 'medium',
            '8. Process, plant and machine operatives': 'skilled',
            '9. Elementary occupations': 'skilled'},
        segment_to_id_mapping={
            'higher': 1,
            'medium': 2,
            'skilled': 3})
    _QS606_non_work = actual_NTEM_population.loc[actual_NTEM_population["e"] >= 3].rename(columns={"C_NTEM": "Persons"})
    _QS606_non_work = _QS606_non_work.groupby(['z', 'd', 'r'], as_index=False)["Persons"].sum().assign(s=4)
    _QS606 = pd.concat([_QS606_work, _QS606_non_work]).sort_values(by=["z", "s"]).reset_index(drop=True)

    _QS609 = _segment_qs(
        census_QS=QS609,
        NTEM_population=actual_NTEM_population,
        segment_letter='n',
        column_to_segment_mapping={
           '1. Higher managerial, administrative and professional occupations': 'NS-SeC 1-2',
           '2. Lower managerial, administrative and professional occupations': 'NS-SeC 1-2',
           '3. Intermediate occupations': 'NS-SeC 3-5',
           '4. Small employers and own account workers': 'NS-SeC 3-5',
           '5. Lower supervisory and technical occupations': 'NS-SeC 3-5',
           '6. Semi-routine occupations': 'NS-SeC 6-7',
           '7. Routine occupations': 'NS-SeC 6-7',
           '8. Never worked and long-term unemployed': 'NS-SeC 8',
           'L15 Full-time students': 'L15'},
        segment_to_id_mapping={
            'NS-SeC 1-2': 1,
            'NS-SeC 3-5': 2,
            'NS-SeC 6-7': 3,
            'NS-SeC 8': 4,
            'L15': 5})

    _write_ipfn_inputs(seed_population=seed_population,
                       QS401=_QS401,
                       QS606=_QS606,
                       QS609=_QS609,
                       district_count=lookup_dict["geography"]["d"].max())

    census_and_by_lu_obj.state['3.1.2 expand population segmentation'] = 1
    logging.info('3.1.2 expand population segmentation completed')
    return None


# Define function that can be used to get 2011 NTEM data
def ntem_pop_interpolation(census_and_by_lu_obj):
    """
    Process population data from NTEM CTripEnd database:
    Interpolate population to the target year, in this case, it is for base year 2011 as databases
    are available in 5 year interval;
    Translate NTEM zones in Scotland into NorNITs zones; for England and Wales, NTEM zones = NorMITs zones (MSOAs)
    """

    # The year of data is set to define the upper and lower NTEM run years and interpolate as necessary between them.
    # The base year for NTEM is 2011 and it is run in 5-year increments from 2011 to 2051.
    # The year selected below must be between 2011 and 2051 (inclusive).
    Year = 2011

    if Year < 2011 | Year > 2051:
        raise ValueError("Please select a valid year of data.")
    else:
        pass

    Output_Folder = census_and_by_lu_obj.home_folder + '/Outputs/'
    print(Output_Folder)
    LogFile = Output_Folder + 'LogFile.txt'
    # 'I:/NorMITs Synthesiser/Zone Translation/'
    Zone_path = census_and_by_lu_obj.zones_folder + 'Export/ntem_to_msoa/ntem_msoa_pop_weighted_lookup.csv'
    Pop_Segmentation_path = census_and_by_lu_obj.import_folder + 'CTripEnd/Pop_Segmentations.csv'
    with open(LogFile, 'w') as o:
        o.write("Notebook run on - " + str(datetime.datetime.now()) + "\n")
        o.write("\n")
        o.write("Data Year - " + str(Year) + "\n")
        o.write("\n")
        o.write("Correspondence Lists:\n")
        o.write(Zone_path + "\n")
        o.write(Pop_Segmentation_path + "\n")
        o.write("\n")

    # Data years
    # NTEM is run in 5-year increments with a base of 2011.
    # This section calculates the upper and lower years of data that are required
    InterpolationYears = Year % 5
    LowerYear = Year - ((InterpolationYears - 1) % 5)
    UpperYear = Year + ((1 - InterpolationYears) % 5)

    print("Lower Interpolation Year - " + str(LowerYear))
    print("Upper Interpolation Year - " + str(UpperYear))

    # Import Upper and Lower Year Tables
    # 'I:/Data/NTEM/NTEM 7.2 outputs for TfN/'
    # TODO: Strict wrappers around DB calls, very likely to shift
    # TODO: Also need an alternative seed source to be specified upstream (e.g. TEMPro, Addressbase)

    LowerNTEMDatabase = census_and_by_lu_obj.CTripEnd_Database_path + 'CTripEnd7_' + str(LowerYear) + '.accdb'
    UpperNTEMDatabase = census_and_by_lu_obj.CTripEnd_Database_path + 'CTripEnd7_' + str(UpperYear) + '.accdb'
    # UpperNTEMDatabase = census_and_by_lu_obj.CTripEnd_Database_path + r"\CTripEnd7_" + str(UpperYear) + r".accdb"
    cnxn = pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + \
                          '{};'.format(UpperNTEMDatabase))

    query = r"SELECT * FROM ZoneData"
    UZoneData = pd.read_sql(query, cnxn)
    cnxn.close()

    cnxn = pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' +
                          '{};'.format(LowerNTEMDatabase))

    query = r"SELECT * FROM ZoneData"
    LZoneData = pd.read_sql(query, cnxn)
    cnxn.close()

    # Re-format Tables
    LZonePop = LZoneData.copy()
    UZonePop = UZoneData.copy()
    LZonePop.drop(
        ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'K01',
         'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15'], axis=1,
        inplace=True)
    UZonePop.drop(
        ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'K01',
         'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15'], axis=1,
        inplace=True)
    LZonePop_long = pd.melt(LZonePop, id_vars=["I", "R", "B", "Borough", "ZoneID", "ZoneName"],
                            var_name="LTravellerType", value_name="LPopulation")
    UZonePop_long = pd.melt(UZonePop, id_vars=["I", "R", "B", "Borough", "ZoneID", "ZoneName"],
                            var_name="UTravellerType", value_name="UPopulation")

    LZonePop_long.rename(columns={"I": "LZoneID", "B": "LBorough", "R": "LAreaType"}, inplace=True)
    UZonePop_long.rename(columns={"I": "UZoneID", "B": "UBorough", "R": "UAreaType"}, inplace=True)

    LZonePop_long['LIndivID'] = LZonePop_long.LZoneID.map(str) + "_" + LZonePop_long.LAreaType.map(
        str) + "_" + LZonePop_long.LBorough.map(str) + "_" + LZonePop_long.LTravellerType.map(str)
    UZonePop_long['UIndivID'] = UZonePop_long.UZoneID.map(str) + "_" + UZonePop_long.UAreaType.map(
        str) + "_" + UZonePop_long.UBorough.map(str) + "_" + UZonePop_long.UTravellerType.map(str)

    # Join Upper and Lower Tables
    TZonePop_DataYear = LZonePop_long.join(UZonePop_long.set_index('UIndivID'), on='LIndivID', how='right',
                                           lsuffix='_left', rsuffix='_right')
    TZonePop_DataYear.drop(['UZoneID', 'UBorough', 'UAreaType', 'UTravellerType'], axis=1, inplace=True)

    # Interpolate Between Upper and Lower Years
    TZonePop_DataYear['GrowthinPeriod'] = TZonePop_DataYear.eval('UPopulation - LPopulation')
    TZonePop_DataYear['GrowthperYear'] = TZonePop_DataYear.eval('GrowthinPeriod / 5')
    TZonePop_DataYear = TZonePop_DataYear.assign(GrowthtoYear=TZonePop_DataYear['GrowthperYear'] * (Year - LowerYear))
    TZonePop_DataYear['Population'] = TZonePop_DataYear.eval('LPopulation + GrowthtoYear')

    # Tidy up
    TZonePop_DataYear.rename(
        columns={"LZoneID": "ZoneID", "LBorough": "Borough", "LAreaType": "AreaType", "LTravellerType": "TravellerType",
                 "LIndivID": "IndivID"}, inplace=True)
    TZonePop_DataYear.drop(
        ['GrowthinPeriod', 'GrowthperYear', 'GrowthtoYear', 'LPopulation', 'UPopulation', 'ZoneID_left', 'ZoneID_right',
         'ZoneName_right', 'ZoneName_left', 'Borough_left', 'Borough_right', 'IndivID'], axis=1, inplace=True)
    print(TZonePop_DataYear.Population.sum())

    # Translating zones for those in Scotland
    Zone_List = pd.read_csv(Zone_path)
    TZonePop_DataYear = TZonePop_DataYear.join(Zone_List.set_index('ntemZoneID'), on='ZoneID', how='right')
    # TZonePop_DataYear.rename(columns={'msoaZoneID': 'ModelZone'}, inplace=True)
    TZonePop_DataYear[
        'Population_RePropped'] = TZonePop_DataYear['Population'] * TZonePop_DataYear['overlap_ntem_pop_split_factor']

    Segmentation_List = pd.read_csv(Pop_Segmentation_path)
    TZonePop_DataYear = TZonePop_DataYear.join(Segmentation_List.set_index('NTEM_Traveller_Type'), on='TravellerType',
                                               how='right')
    TZonePop_DataYear.drop(
        ['Population', 'ZoneID', 'overlap_population', 'ntem_population', 'msoa_population',
         'overlap_msoa_pop_split_factor', 'overlap_type'], axis=1, inplace=True)
    TZonePop_DataYear.rename(columns={"Population_RePropped": "Population"}, inplace=True)
    print(TZonePop_DataYear.Population.sum())
    TZonePop_DataYear = TZonePop_DataYear.groupby(['msoaZoneID', 'AreaType', 'Borough', 'TravellerType',
                                                   'NTEM_TT_Name', 'Age_code', 'Age',
                                                   'Gender_code', 'Gender', 'Household_composition_code',
                                                   'Household_size', 'Household_car', 'Employment_type_code',
                                                   'Employment_type'])[
        ['Population']].sum().reset_index()
    NTEM_HHpop = TZonePop_DataYear
    # Export
    Export_SummaryPop = TZonePop_DataYear.groupby(['TravellerType', 'NTEM_TT_Name']).sum()
    print(Export_SummaryPop.Population.sum())
    # Export_SummaryPop.drop(['msoaZoneID'], inplace=True, axis=1)
    PopOutput = "NTEM_{}_Population.csv".format(Year)

    with open(Output_Folder + PopOutput, "w", newline='') as f:
        TZonePop_DataYear.to_csv(f, header=True, sep=",")
    f.close()

    with open(LogFile, "a") as o:
        o.write("Total Population: \n")
        Export_SummaryPop.to_csv(o, header=False, sep="-")
        o.write("\n")
        o.write("\n")
    print("Export complete.")
    print(NTEM_HHpop.head(5))

    census_and_by_lu_obj.state['3.1.1 derive 2011 population from NTEM and convert Scottish zones'] = 1
    logging.info('3.1.1 derive 2011 population from NTEM and convert Scottish zones completed')

    return NTEM_HHpop


def ipf_process(
        seed_path,
        ctrl_NTEM_p_path,
        ctrl_dt_p_path,
        ctrl_NSSEC_p_path,
        ctrl_SOC_p_path,
        output_path
):
    seed = pd.read_csv(seed_path)
    # seed_cols = ['z', 'a', 'g', 'h', 'e', 't', 'n', 's', 'population']
    # seed = seed[seed_cols]
    ctrl_NTEM_p = pd.read_csv(ctrl_NTEM_p_path)
    ctrl_dt_p = pd.read_csv(ctrl_dt_p_path)
    ctrl_NSSEC_p = pd.read_csv(ctrl_NSSEC_p_path)
    ctrl_SOC_p = pd.read_csv(ctrl_SOC_p_path)
    # convert heading on population to total
    # prepare marginals
    seed.rename(columns={"population": "total"}, inplace=True)
    ctrl_NTEM_p.rename(columns={"population": "total"}, inplace=True)
    ctrl_dt_p.rename(columns={"Persons": "total"}, inplace=True)
    ctrl_NSSEC_p.rename(columns={"Persons": "total"}, inplace=True)
    ctrl_SOC_p.rename(columns={"Persons": "total"}, inplace=True)
    ctrl_NTEM_p = ctrl_NTEM_p.groupby(['z', 'ntem_tt'])['total'].sum()
    ctrl_dt_p = ctrl_dt_p.groupby(['z', 't'])['total'].sum()
    ctrl_NSSEC_p = ctrl_NSSEC_p.groupby(['z', 'n'])['total'].sum()
    ctrl_SOC_p = ctrl_SOC_p.groupby(['z', 's'])['total'].sum()
    marginals = [ctrl_NTEM_p, ctrl_dt_p, ctrl_NSSEC_p,  ctrl_SOC_p]
    furnessed_df, iters, conv = ipfn.ipf_dataframe(
        seed_df=seed,
        target_marginals=marginals,
        value_col="total",
        max_iterations=5000,
        tol=1e-9,
        min_tol_rate=1e-9,
        show_pbar=False)
    # logging.info('The iteration and convergence for district ' + dist_str + ' is: ', iters, conv)
    furnessed_df.to_csv(output_path, index=False)


def ipfn_process_2011(census_and_by_lu_obj):
    # TODO: Is this slow or not?
    """
    Reads in the district chunked IPFN seed and control files
    and creates a compressed file output containing final 2011 f by z, a, g, h, e, t, n, s
    !!!!! IMPORTANT NOTE !!!!! - ART, 28/10/2021 - I think this has now been resolved & it only takes 20 mins to write
    This script was NOT how the IPFN process was actually run
    In order for it to complete in a timely manner, 16 Jupyter Notebooks
    containing the script were run in parallel, each handling up to 20
    districts, even then total tun time on the longest running workbook
    was OVER 24 HOURS. Should anyone ever wish to rerun this process, it
    is recommended that the jupyter notebooks are run in parallel again,
    although this script will still produce the outputs. However, it will
    do it one at a time, so COULD TAKE UP TO A FORTNIGHT!
    The jupyter notebooks that were actually used can be found here:
    I:/NorMITs Land Use/import/2011 Census Furness/02 Process
    """
    # Set min and max districts to process
    # Note that these should be in range 1 to 313 inclusive
    # Setting range to 1 -> 313 inclusive will run ipfn on everything
    # In reality, use the Jupyter notebooks in parallel as mentioned above

    start = 1
    end = 313
    # Output_Folder = r'I:\NorMITs Land Use\import\2011 Census Furness\03 Output'
    # seed_path_start = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\01 Seed Files\2011_NorMITs_seed_d'
    # ctrl_NTEM_p_path_start = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\02 Ctrl1 NTEM Control Files\2011_NorMITs_Ctrl1_NTEM_d'
    # ctrl_dt_p_path_start = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\05 DT Control Files\2011_NorMITs_Ctrl_DT_d'
    # ctrl_NSSEC_p_path_start = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\04 NSSEC Control Files\2011_NorMITs_Ctrl_NSSEC_d'
    # ctrl_SOC_p_path_start = r'I:\NorMITs Land Use\import\2011 Census Furness\01 Inputs\03 SOC Control Files\2011_NorMITs_Ctrl_SOC_d'

    # # Run through the districts specified above in a for loop
    # # Note the 3 ipfn process in each iteration of the loop due to the 3 dimensions that need fitting
    # # TODO: Multi-processing
    # # TODO: Replace with native numpy ndim ipfn
    # # TODO: Psketti code
    # kwarg_list = list()
    # for district_num in range(start, end+1):
    #     dist_str = str(district_num)
    #     print('Working on District', dist_str)
    #     # ----- Format input data ------
    #     seed_path = '_'.join([seed_path_start, dist_str, 'v0.1.csv'])
    #     ctrl_NTEM_p_path = '_'.join([ctrl_NTEM_p_path_start, dist_str, 'v0.1.csv'])
    #     ctrl_dt_p_path = '_'.join([ctrl_dt_p_path_start, dist_str, 'v0.1.csv'])
    #     ctrl_NSSEC_p_path = '_'.join([ctrl_NSSEC_p_path_start, dist_str, 'v0.1.csv'])
    #     ctrl_SOC_p_path = '_'.join([ctrl_SOC_p_path_start, dist_str, 'v0.1.csv'])
    #     output_filename = '_'.join([dist_str, 'furnessed_2011Pop.csv'])
    #     output_path = os.path.join(Output_Folder, output_filename)
    #
    #     kwarg_list.append({
    #         "seed_path": seed_path,
    #         "ctrl_NTEM_p_path": ctrl_NTEM_p_path,
    #         "ctrl_dt_p_path": ctrl_dt_p_path,
    #         "ctrl_NSSEC_p_path": ctrl_NSSEC_p_path,
    #         "ctrl_SOC_p_path": ctrl_SOC_p_path,
    #         "output_path": output_path,
    #     })
    #
    # concurrency.multiprocess(
    #     fn=ipf_process,
    #     kwarg_list=kwarg_list,
    #     pbar_kwargs={"disable": False}
    # )

    # Read the IPFN output files back in,
    # processes them into a single df and calculates f for 2011,
    # then saves out a compressed file with the results

    # Set read in/out variables
    lookup_directory = r'I:\NorMITs Land Use\import\2011 Census Furness\04 Post processing\Lookups'
    z2d2r_lookup_file = r'lookup_geography_z2d2r.csv'
    n_t_ntemtt_lookup_file = r'lookup_ntemtt_to_aghe.csv'
    n_t_ntemtt_lookup = pd.read_csv(os.path.join(lookup_directory, n_t_ntemtt_lookup_file))
    z2d2r_lookup = pd.read_csv(os.path.join(lookup_directory, z2d2r_lookup_file))[['z', 'd']]
    output_directory = r'I:\NorMITs Land Use\import\2011 Census Furness\04 Post processing\Outputs'
    model_name = 'NorMITs'
    model_year = '2011'
    input_directory = r'I:\NorMITs Land Use\import\2011 Census Furness\03 Output\\'
    input_filename = r'_furnessed_2011Pop.csv'
    # Edit these parameters to only read in part of the dataset as a test


    list_of_df = []
    # Loop over all districts and append to a master df.
    for district in range(start, end+1):
        dist_str = str(district)
        input_path = [input_directory, dist_str, input_filename]
        input_df = pd.read_csv(''.join(input_path))
        list_of_df.append(input_df)
    print('All', str(end), 'files read in ok')
    furnessed_df = pd.concat(list_of_df, axis=0, ignore_index=True)

    # Process to get f
    # Join the n_t_ntem_tt lookup and tidy
    expanded_furnessed_df = furnessed_df.copy()
    expanded_furnessed_df = pd.merge(expanded_furnessed_df,
                                     n_t_ntemtt_lookup,
                                     how='left',
                                     on='ntem_tt')
    expanded_furnessed_df = pd.merge(expanded_furnessed_df,
                                     z2d2r_lookup,
                                     how='left',
                                     on='z')
    # Create P grouped by zone and a, g, h, e
    # Then reapply this to the main df and tidy
    grouped_P_df = expanded_furnessed_df.groupby(['z', 'a', 'g', 'h', 'e']
                                                 )['total'].sum().reset_index()
    grouped_P_df = grouped_P_df.rename(columns={'total': 'P_zaghe'})
    expanded_furnessed_df = pd.merge(expanded_furnessed_df,
                                     grouped_P_df,
                                     how='left',
                                     on=['z', 'a', 'g', 'h', 'e'])
    expanded_furnessed_df = expanded_furnessed_df.rename(
        columns={'total': 'P_zaghetns'})
    # Calculate f
    expanded_furnessed_df['f_tns|zaghe'] = expanded_furnessed_df['P_zaghetns'] / expanded_furnessed_df['P_zaghe']

    # Assign f to empty categories
    # Get these f's from the district averages for unique a, g, h, e, t, n, s combinations
    grouped_P_by_d_df = expanded_furnessed_df.groupby(['d', 'a', 'g', 'h', 'e'])['P_zaghetns'].sum().reset_index()
    grouped_P_by_d_df = grouped_P_by_d_df.rename(columns={'P_zaghetns': 'P_daghe'})
    expanded_furnessed_df = pd.merge(expanded_furnessed_df,
                                     grouped_P_by_d_df,
                                     how='left',
                                     on=['d', 'a', 'g', 'h', 'e'])
    grouped_P_by_d_all_df = expanded_furnessed_df.groupby(['d', 'a', 'g', 'h', 'e', 't', 'n', 's'])[
        'P_zaghetns'].sum().reset_index()
    grouped_P_by_d_all_df = grouped_P_by_d_all_df.rename(columns={'P_zaghetns': 'P_daghetns'})
    expanded_furnessed_df = pd.merge(expanded_furnessed_df,
                                     grouped_P_by_d_all_df,
                                     how='left',
                                     on=['d', 'a', 'g', 'h', 'e', 't', 'n', 's'])
    expanded_furnessed_df['f_tnsdaghe'] = expanded_furnessed_df['P_daghetns'] / expanded_furnessed_df['P_daghe']
    # fill in f nans at the zonal level with district level f's
    expanded_furnessed_df = expanded_furnessed_df.rename(columns={'f_tns|zaghe': 'f_tnszaghe'})
    expanded_furnessed_df.f_tnszaghe.fillna(expanded_furnessed_df.f_tnsdaghe, inplace=True)
    expanded_furnessed_df = expanded_furnessed_df.rename(
        columns={'f_tnszaghe': 'f_tns|zaghe', 'f_tnsdaghe': 'f_tns|daghe'})

    # Do some checks
    print('Shape of expanded furnessed df is', expanded_furnessed_df.shape)
    check_f_by_zaghe = expanded_furnessed_df.groupby(['z'])['f_tns|zaghe'].sum().reset_index()
    print('Max f by z is:', str(check_f_by_zaghe['f_tns|zaghe'].max()))
    print('Min f by z is:', str(check_f_by_zaghe['f_tns|zaghe'].min()))
    check_2011_f_filename = '_'.join([model_name, model_year, 'check_sum_of_f_by_z_with_d_level_f_fills.csv'])
    check_f_by_zaghe.to_csv(check_2011_f_filename, index=False)
    print('Population in output dataframe is', expanded_furnessed_df['P_zaghetns'].sum())

    # Write out files.
    # Use the following two line to control which outputs to write out
    # Set the variables to 0 to stop write out and 1 to keep it.
    # TODO: Optimise writes
    write_out_f = 1
    write_out_P = 1
    if write_out_f == 1:
        expanded_furnessed_f_df_out = expanded_furnessed_df[['z', 'a', 'g', 'h', 'e', 't', 'n', 's', 'f_tns|zaghe']]
        output_file_name = '_'.join([model_name, model_year, 'post_ipfn_f_values.csv'])
        output_file_path = os.path.join(output_directory, output_file_name)
        print('Starting to print output file at:', datetime.datetime.now())
        print('Hang tight, this could take a while!')
        expanded_furnessed_f_df_out.to_csv(output_file_path, index=False)
        print('Printing output file complete at:', datetime.datetime.now())
    if write_out_P == 1:
        expanded_furnessed_P_df_out = expanded_furnessed_df[['z', 'a', 'g', 'h', 'e', 't', 'n', 's', 'P_zaghetns']]
        output_file_name = '_'.join([model_name, model_year, 'post_ipfn_P_values.csv'])
        output_file_path = os.path.join(output_directory, output_file_name)
        print('Starting to print output file at:', datetime.datetime.now())
        print('Hang tight, this could take a while!')
        expanded_furnessed_P_df_out.to_csv(output_file_path, index=False)
        print('Printing output file complete at:', datetime.datetime.now())

    census_and_by_lu_obj.state['3.1.3 data synthesis'] = 1
    logging.info('3.1.3 data synthesis completed')
