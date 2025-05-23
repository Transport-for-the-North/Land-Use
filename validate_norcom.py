from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import yaml
from caf.base import DVector

from land_use.norcom import NorCOMResult, Params, apply_norcom
from land_use.data_processing import read_dvector_data, write_to_excel
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

# infill factor in case of negative households in a zone
fudge = 0.5

# 0v1+ Model Correction (applied relative to 0 cars category)
# 1v2+ Model Correction (applied relative to the 1 car category)
adjustment_factors = {
    'NE': {"0_v_1+": -0.05, "1_v_2+": -0.265},
    'NW': {"0_v_1+": -0.075, "1_v_2+": -0.25},
    'YH': {"0_v_1+": -0.055, "1_v_2+": -0.245},
    'Wales': {"0_v_1+": -0.07, "1_v_2+": -0.3},
    'WM': {"0_v_1+": -0.08, "1_v_2+": -0.27},
    'EM': {"0_v_1+": -0.05, "1_v_2+": -0.25},
    'SW': {"0_v_1+": -0.07, "1_v_2+": -0.23},
    'EoE': {"0_v_1+": -0.065, "1_v_2+": -0.2},
    'Lon': {"0_v_1+": -0.015, "1_v_2+": -0.23},
    'SE': {"0_v_1+": -0.06, "1_v_2+": -0.2},
}

result_dict = {}
all_applied, all_expected = [], []
# loop through regions
for GOR in GORS:

    # define path to region specific input dvector
    input_dvector = params.input_dvectors / f'Output P{params.file_reference}_{GOR}.hdf'
    # load the 2021 household output that we are trying to validate
    input_data = DVector.load(input_dvector)

    # apply norcom with adjustments
    adjusted_norcom = apply_norcom(
        any_car_ownership_result=any_car_ownership,
        multiple_car_ownership_result=multiple_car_ownership,
        input_dvector=input_data,
        any_car_ownership_correction=adjustment_factors.get(GOR).get("0_v_1+"),
        multiple_car_ownership_correction=adjustment_factors.get(GOR).get("1_v_2+")
    )

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
    adjusted_norcom.aggregate(['car_availability']).save(data_dir / f'applied_{GOR}.hdf')
    validation_data.save(data_dir / f'expected_{GOR}.hdf')

    # calculate DVectors of differences
    absolute = adjusted_norcom.aggregate(['car_availability']) - validation_data
    incremental = adjusted_norcom.aggregate(['car_availability']) / validation_data
    absolute.save(data_dir / f'absolute_{GOR}.hdf')
    incremental.save(data_dir / f'incremental_{GOR}.hdf')

    # add to result dictionnaries to output
    result_dict[f'{GOR}_APPLIED'] = adjusted_norcom.aggregate(['car_availability']).data.reset_index().melt(
        id_vars=['car_availability'], value_vars=adjusted_norcom.data.columns,
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
