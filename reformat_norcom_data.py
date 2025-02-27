from pathlib import Path
from itertools import product

import pandas as pd

from land_use.preprocessing import (
    reduce_classified_build, convert_price_base, NORCOM_MAPPINGS, NORCOM_BANDINGS
)

# define path of main working directory to provide inputs to running norcom
working_dir = Path(r'I:\NorMITs NorCOM\Import')

# define folder to write main NTS based inputs for use in NorCOM
output_folder = working_dir / 'NTS'

# define folder of price deflation inputs to convert NTS price data from
# 2002 to 2023
price_deflator = pd.read_csv(working_dir / 'RPI' / 'rpi.csv')

# define folder of car cost inputs
car_cost = pd.read_csv(working_dir / 'Car Cost' / 'car_cost.csv')

# define folder of GDP inputs
gdp = pd.read_csv(working_dir / 'GDP' / 'nts_to_gdp_correspondence.csv')

# --- collapse the classified build to unique household attributes only --- #
# define path to classified build, its assumed this is a standard output of
# tfns internal processes. This is assumed to be a trip level dataset.
input_classified_build = Path(r'I:\NTS\classified builds\cb_tfn_v2023.1.csv')

# read in the data
classified_build = pd.read_csv(input_classified_build)

# drop duplicates over the trip level data on household attributes
nts_hh_data = reduce_classified_build(
    trip_data=classified_build
)

# apply rpi based adjustment factor to convert NTS nominal prices to 2023
nts_hh_data = convert_price_base(
    data=nts_hh_data, deflator=price_deflator, index_column='2023_deflator'
)

# melt the gdp data to long format to merge with nts
gdp = pd.melt(
    gdp, id_vars=['hholdoslaua_b01id'], var_name='surveyyear', value_name='gdp_pc'
)
gdp['surveyyear'] = gdp['surveyyear'].astype(int)

# calculate gdp deflator in 2023 base to apply to car cost columns
_2023 = gdp[gdp['surveyyear'] == 2023].rename(columns={'gdp_pc': 'gdp_2023'})
gdp = gdp.merge(_2023[['hholdoslaua_b01id', 'gdp_2023']], on='hholdoslaua_b01id', how='left')
gdp['gdp_deflator'] = gdp['gdp_pc'] / gdp['gdp_2023']

# merge the running and purchase costs, and gdp information to NTS data
merged_data = nts_hh_data.merge(
    car_cost, on='surveyyear', how='left'
).merge(
    gdp, on=['hholdoslaua_b01id', 'surveyyear'], how='left'
)

# apply gdp deflator to car cost columns
for col in merged_data.columns:
    if col.endswith('_cost'):
        merged_data[f'deflated_{col}'] = merged_data[col] * merged_data['gdp_deflator']

# --- define new columns based on aggregations of other columns --- #
# mappings are 1 to 1 lookup dictionary mappings
for new_col, mapping in NORCOM_MAPPINGS.items():
    for original_col, mapper in mapping.items():
        merged_data[new_col] = merged_data[original_col].map(mapper)
        print(merged_data[new_col].value_counts())

# bandings are pd.cut bandings
for new_col, banding in NORCOM_BANDINGS.items():
    for original_col, bander in banding.items():
        merged_data[new_col] = pd.cut(
            merged_data[original_col], bander.cut_off_values, labels=bander.labels, right=False
        )
        print(merged_data[new_col].value_counts())

# TODO: automate this a bit, just doing something quick to test interaction terms
for i, j in product(range(1, 6), range(1, 4)):
    pt_mapping = {
        (i, j): 1
    }
    merged_data[f'walkrail{i}_desc{j}'] = (
            merged_data['walkrail_b01id'].eq(i) * merged_data['descta_b01id'].eq(j)
    ).fillna(0).astype(int)
    merged_data[f'walkrail{i}_desc{j}'] = merged_data[f'walkrail{i}_desc{j}'].where(
        merged_data['walkrail_b01id'].ne(-10), -10
    )


# write household data to working folder
merged_data.to_csv(output_folder / 'nts_hh_data_v2.csv', index=False)
