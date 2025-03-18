from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import yaml
from caf.base import DVector

from land_use.norcom import NorCOMResult
from land_use.data_processing import create_dvector_from_data, read_dvector_data, write_to_excel
from land_use.constants import GORS, LSOA_NAME

# TODO: expand on the documentation here
parser = ArgumentParser('Land-Use NorCOM validation command line runner')
parser.add_argument('config_file', type=Path)
args = parser.parse_args()

# load configuration file
with open(args.config_file, 'r') as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

# Set up inputs from yaml
estimation_version = str(config['estimation version'])
results_path = Path(config['results path'])
zonal_lookups = Path(config['zonal lookups'])
input_dvectors = Path(config['input dvectors'])
validation_dvector = Path(config['validation dvector'])
output_path = Path(config['output path'])

# create output folder
OUTPUT_DIR = output_path / estimation_version
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# load the NorCOM results
any_car_ownership = NorCOMResult.from_coefficients_csv(
    csv_path=results_path / estimation_version / '0v1+' / 'output' / 'final_model_coefficients.csv',
    case_category='1+', noncase_category='0',
    zonal_lookups= zonal_lookups / f'zonal_logit_data_{estimation_version}.csv'
)

multiple_car_ownership = NorCOMResult.from_coefficients_csv(
    csv_path=results_path / estimation_version / '1v2+' / 'output' / 'final_model_coefficients.csv',
    case_category='2+', noncase_category='1', dependent_category='1+',
    zonal_lookups= zonal_lookups / f'zonal_logit_data_{estimation_version}.csv'
)

# expand the results to have all three probability levels in one dataframe
result = any_car_ownership * multiple_car_ownership

result_dict = {}
all_applied, all_expected = [], []
# loop through regions
for GOR in GORS:
    # convert to DVector with the right zone system subset
    probabilities = create_dvector_from_data(
        dvector_data=result, geographical_level=any_car_ownership.zonal_definition,
        input_segments=list(result.index.names), geography_subset=GOR
    )

    # define path to region specific input dvector
    input_dvector = input_dvectors / f'Output P4.3_{GOR}.hdf'
    # load the 2021 household output that we are trying to validate
    _2021_data = DVector.load(input_dvector).aggregate(['accom_h', 'ns_sec', 'adults', 'children'])
    # apply norcom to this 2021 modelled output
    apply_norcom = _2021_data * probabilities
    # agggregate the post-norcom data to just car availability by zone
    validation = apply_norcom.aggregate(['car_availability'])

    # load the validation DVector, just number of households in each car ownership
    # category by LSOA from the census
    census_data = read_dvector_data(
        file_path=validation_dvector, geographical_level=LSOA_NAME,
        input_segments=['car_availability'], geography_subset=GOR
    )

    # add to result dictionnaries to output
    result_dict[f'{GOR}_APPLIED'] = validation.data.reset_index().melt(
        id_vars=['car_availability'], value_vars=validation.data.columns,
        var_name=LSOA_NAME, value_name='households'
    )
    result_dict[f'{GOR}_EXPECTED'] = census_data.data.reset_index().melt(
        id_vars=['car_availability'], value_vars=census_data.data.columns,
        var_name=LSOA_NAME, value_name='households'
    )
    # create list of all GORs to combine to a single output
    all_applied.append(result_dict[f'{GOR}_APPLIED'])
    all_expected.append(result_dict[f'{GOR}_EXPECTED'])

# create all GOR output
all_applied = pd.concat(all_applied)
all_expected = pd.concat(all_expected)

# write outputs
write_to_excel(
    output_folder=OUTPUT_DIR,
    file='2021_validation.xlsx',
    dfs=[all_applied, all_expected] + list(result_dict.values()),
    sheet_names=['APPLIED', 'EXPECTED'] + list(result_dict.keys())
)
