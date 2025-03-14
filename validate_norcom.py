from pathlib import Path

from caf.base import DVector

from land_use.norcom import NorCOMResult
from land_use.data_processing import create_dvector_from_data, read_dvector_data

# Set up file paths (eventually should be yaml)
estimation_version = 'v36'
results_path = Path(r'I:\NorMITs NorCOM\AECOM working')
zonal_lookups = Path(r'I:\NorMITs NorCOM\Import\Zonal Data')
input_dvector = Path(r'F:\Deliverables\Land-Use\241213_Population\01_Intermediate Files\Output P4.3_NW.hdf')
validation_dvector = Path(r'I:\NorMITs Land Use\2023\import\ONS-validation\preprocessing\households_cars_lsoa_3.hdf')

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

# expand the results to have all three probability levels in one dataframe and
# convert to DVector
result = any_car_ownership * multiple_car_ownership
probabilities = create_dvector_from_data(
    dvector_data=result, geographical_level=any_car_ownership.zonal_definition,
    input_segments=list(result.index.names), geography_subset='NW'
)

# load the 2021 household output that we are trying to validate
_2021_data = DVector.load(input_dvector).aggregate(['accom_h', 'ns_sec', 'adults', 'children'])
# apply norcom to this 2021 modelled output
apply_norcom = _2021_data * probabilities
# agggregate the post-norcom data to just car availability by zone
validation = apply_norcom.aggregate(['car_availability'])

# load the validation DVector, just number of households in each car ownership
# category by LSOA from the census
census_data = read_dvector_data(
    file_path=validation_dvector, geographical_level=any_car_ownership.zonal_definition,
    input_segments=['car_availability'], geography_subset='NW'
)

# compare the two
abs_diff = validation - census_data
perc_diff = (validation / census_data) - 1
