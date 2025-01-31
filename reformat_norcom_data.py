from pathlib import Path

import pandas as pd

from land_use.preprocessing import reduce_classified_build, NORCOM_MAPPINGS


# define folder of main NTS based inputs to NorCOM
output_folder = Path(r'I:\NorMITs NorCOM\Import\NTS')

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

# --- define new columns based on aggregations of other columns --- #
for new_col, mapping in NORCOM_MAPPINGS.items():
    for original_col, mapper in mapping.items():
        data[new_col] = data[original_col].map(mapper)
        print(data[new_col].value_counts())

# write household data to working folder
data.to_csv(output_folder/ 'nts_hh_data.csv', index=False)
