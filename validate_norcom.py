from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import yaml
from caf.base import DVector

from land_use.norcom import NorCOMResult, Params
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
params = Params.from_yaml(config)

# create output folder
OUTPUT_DIR = params.output_path / params.estimation_version / params.year
data_dir = OUTPUT_DIR / 'data'
data_dir.mkdir(exist_ok=True, parents=True)

# load the NorCOM results
any_car_ownership = NorCOMResult.from_coefficients_csv(
    csv_path=params.results_path / params.estimation_version / '0v1+' / 'output' / 'final_model_coefficients.csv',
    case_category='1+', noncase_category='0',
    zonal_lookups=params.zonal_lookups / f'zonal_logit_data_{params.estimation_version}_{params.year}.csv'
)

multiple_car_ownership = NorCOMResult.from_coefficients_csv(
    csv_path=params.results_path / params.estimation_version / '1v2+' / 'output' / 'final_model_coefficients.csv',
    case_category='2+', noncase_category='1', dependent_category='1+',
    zonal_lookups=params.zonal_lookups / f'zonal_logit_data_{params.estimation_version}_{params.year}.csv'
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
    input_dvector = params.input_dvectors / f'Output P{params.file_reference}_{GOR}.hdf'
    # load the 2021 household output that we are trying to validate
    input_data = DVector.load(input_dvector)
    # apply norcom to this modelled output
    apply_norcom = input_data.aggregate(['accom_h', 'ns_sec', 'adults', 'children']) * probabilities
    # aggregate the post-norcom data to just car availability by zone
    validation = apply_norcom.aggregate(['car_availability'])

    # load the validation DVector, just number of households in each car ownership
    # category by LSOA from the census
    if params.validate():
        validation_data = read_dvector_data(
            file_path=params.validation_dvector, geographical_level=LSOA_NAME,
            input_segments=['car_availability'], geography_subset=GOR
        )
    else:
        validation_data = input_data.aggregate(['car_availability'])

    # save DVectors of validation
    validation.save(data_dir / f'applied_{GOR}.hdf')
    validation_data.save(data_dir / f'expected_{GOR}.hdf')

    # calculate DVectors of differences
    absolute = validation - validation_data
    incremental = validation / validation_data
    absolute.save(data_dir / f'absolute_{GOR}.hdf')
    incremental.save(data_dir / f'incremental_{GOR}.hdf')

    # add to result dictionnaries to output
    result_dict[f'{GOR}_APPLIED'] = validation.data.reset_index().melt(
        id_vars=['car_availability'], value_vars=validation.data.columns,
        var_name=LSOA_NAME, value_name='households'
    )
    result_dict[f'{GOR}_EXPECTED'] = validation_data.data.reset_index().melt(
        id_vars=['car_availability'], value_vars=validation_data.data.columns,
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
    file=f'{params.year}_validation.xlsx',
    dfs=[all_applied, all_expected] + list(result_dict.values()),
    sheet_names=['APPLIED', 'EXPECTED'] + list(result_dict.keys())
)
