from pathlib import Path

import pandas as pd

from land_use.preprocessing import (
    reduce_classified_build, convert_price_base, NORCOM_MAPPINGS
)
from land_use.constants import (
    _MODEL_PRICE_BASE, _NTS_PRICE_BASE
)

# define path of main working directory to provide inputs to running norcom
working_dir = Path(r'I:\NorMITs NorCOM\Import')

# define folder of main NTS based inputs to NorCOM
nts_folder = working_dir / 'NTS'

# define folder of price deflation inputs to convert NTS price data from
# 2002 to 2023
price_deflator = working_dir / 'RPI' / 'rpi.csv'
price_deflator = pd.read_csv(price_deflator)

# define folder of car cost inputs
car_cost = working_dir / 'Car Cost' / 'car_cost.csv'
car_cost = pd.read_csv(car_cost)

# --- collapse the classified build to unique household attributes only --- #
# define path to classified build, its assumed this is a standard output of
# tfns internal processes. This is assumed to be a trip level dataset.
classified_build = Path(r'I:\NTS\classified builds\cb_tfn_v2023.1.csv')

# read in the data
data = pd.read_csv(classified_build)

# drop duplicates over the trip level data on household attributes
data = reduce_classified_build(
    trip_data=data
)

# TODO is this needed?
# calculate rpi based adjustment factor to convert prices between 2002 and 2023
factor = (
    price_deflator[price_deflator['surveyyear'].eq(_MODEL_PRICE_BASE)].agg({'rpi': 'sum'}) /
    price_deflator[price_deflator['surveyyear'].eq(_NTS_PRICE_BASE)].agg({'rpi': 'sum'})
).sum()

# convert hh income to 2023 price base year
data[f'hh_income_{_MODEL_PRICE_BASE}'] = (
    data['hh_income'] * factor
) * data['hh_income'].ne(-1) - data['hh_income'].eq(-1)


# attach running and purchase costs to NTS data
data = pd.merge(
    data, car_cost, on='surveyyear', how='left'
)

# --- define new columns based on aggregations of other columns --- #
for new_col, mapping in NORCOM_MAPPINGS.items():
    for original_col, mapper in mapping.items():
        data[new_col] = data[original_col].map(mapper)
        print(data[new_col].value_counts())

# write household data to working folder
data.to_csv(nts_folder/ 'nts_hh_data.csv', index=False)
