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

# 0v1+ Model Correction (applied relative to 0 cars category)
_0_v_1 = -0.1

# 1v2+ Model Correction (applied relative to the 1 car category)
_1_v_2 = -0.2

# infill factor in case of negative households in a zone
fudge = 0.5

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

    # TODO some discussion on caf.base about translate_segments etc. Have asked for a NORCOM_0V1 segment to be added so this can be translated in proper DVector format, current merge doesn't let me do the next line so have commented on the PR
    # model_0v1 = validation.translate_segment('car_availability', 'norcom_0v1+')

    # TODO basically all of this now uses pandas stuff when it should be DVector
    # get results of 0v1+ model
    model_0v1 = validation.data.reset_index()
    model_0v1['norcom_0v1+'] = model_0v1['car_availability'].map({1: 1, 2: 2, 3: 2})
    model_0v1 = model_0v1.groupby('norcom_0v1+').sum().drop(columns=['car_availability'])
    model_0v1 = model_0v1.reset_index().melt(
        id_vars='norcom_0v1+', value_name='households'
    ).pivot(
        index=LSOA_NAME, columns='norcom_0v1+', values='households'
    ).reset_index()
    model_0v1.columns = [LSOA_NAME, '0_cars', '1+_cars']

    # get results of 1v2+ model
    model_1v2 = validation.data.filter(items=[2, 3], axis=0).reset_index().melt(
        id_vars='index', value_name='households'
    ).pivot(
        index=LSOA_NAME, columns='index', values='households'
    ).reset_index()
    model_1v2.columns = [LSOA_NAME, '1_cars', '2+_cars']

    # combine the two models as in the spreadsheet example
    combo = pd.merge(model_0v1, model_1v2, on=LSOA_NAME, how='left')

    # step 1: shift households between 0and1+ model, infill negatives with fudge
    combo['0_cars_step1'] = combo['0_cars'] + (_0_v_1 * combo['1+_cars'])
    combo['0_cars_step1'] = combo['0_cars_step1'].where(combo['0_cars_step1'] > 0, fudge * combo['0_cars'])
    combo['1+_cars_step1'] = combo['1+_cars'] - (_0_v_1 * combo['1+_cars'])

    # step 2: calculate new 1v2+ car numbers based on expected proportions of 1v2+ from the original model
    combo['1_car_step2'] = combo['1+_cars_step1'] * (combo['1_cars'] / (combo['1_cars'] + combo['2+_cars']))
    combo['2+_cars_step2'] = combo['1+_cars_step1'] * (combo['2+_cars'] / (combo['1_cars'] + combo['2+_cars']))

    # step 3: shift households between 1and2+ model, infill negatives with fudge
    combo['1_car_step3'] = combo['1_car_step2'] + (_1_v_2 * combo['2+_cars_step2'])
    combo['1_car_step3'] = combo['1_car_step3'].where(combo['1_car_step3'] > 0, fudge * combo['1_car_step2'])
    combo['2+_cars_step3'] = combo['2+_cars_step2'] - (_1_v_2 * combo['2+_cars_step2'])

    # step 4: isolate the required outputs for 0, 1, 2+ cars
    combo[1] = combo['0_cars_step1']
    combo[2] = combo['1_car_step3']
    combo[3] = combo['2+_cars_step3']

    # convert back to DVector
    adjusted_norcom = combo.melt(
        id_vars=LSOA_NAME, value_vars=[1, 2, 3], value_name='households', var_name='car_availability'
    ).pivot(
        index='car_availability', columns=LSOA_NAME, values='households'
    )
    adjusted_norcom = create_dvector_from_data(
        dvector_data=adjusted_norcom, geographical_level=any_car_ownership.zonal_definition,
        input_segments=['car_availability'], geography_subset=GOR
    )

    # TODO need to reapply the splits of the household segments ['accom_h', 'ns_sec', 'adults', 'children'] to get back to a fully segmented households dataset

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
    adjusted_norcom.save(data_dir / f'applied_{GOR}.hdf')
    validation_data.save(data_dir / f'expected_{GOR}.hdf')

    # calculate DVectors of differences
    absolute = adjusted_norcom - validation_data
    incremental = adjusted_norcom / validation_data
    absolute.save(data_dir / f'absolute_{GOR}.hdf')
    incremental.save(data_dir / f'incremental_{GOR}.hdf')

    # add to result dictionnaries to output
    result_dict[f'{GOR}_APPLIED'] = adjusted_norcom.data.reset_index().melt(
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
