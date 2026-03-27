from functools import reduce
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from caf.base import DVector

from land_use.data_processing import create_dvector_from_data, apply_proportions


# data class for reading and storing NorCOM estimation results
class NorCOMResult:

    def __init__(self, coefficients: pd.DataFrame, case_category: str, noncase_category: str, zonal_lookups: Path,
                 dependent_category: Optional[str] = None, continuous_params: Optional[pd.DataFrame] = None):
        # Store info about the results
        self.case_category = self.prob_col(case_category)
        self.noncase_category = self.prob_col(noncase_category)
        # Only figure out the dependent column if we actually have a dependent category
        self.dependent_category = dependent_category and self.prob_col(dependent_category)

        # Go from individual coefficients to all possible combinations, then add in zones
        self._combinations = self.coefficients_to_combinations(coefficients)
        self.zonal_definition, self._expanded_to_zones = self.expand_to_zones(
            self._combinations, continuous_params, zonal_lookups
        )

        # Figure out the actual probabilities for the case and noncase options
        self.probabilities = self._expanded_to_zones.copy()
        self.probabilities[self.case_category] = 1 / self.probabilities['Combined Coefficient'].add(1)
        self.probabilities[self.noncase_category] = 1 - self.probabilities[self.case_category]

        self.probabilities.drop(
            columns=[c for c in self.probabilities.columns if (c.startswith('Coeff_') or c == 'Combined Coefficient')],
            inplace=True
        )

    @staticmethod
    def prob_col(category: str) -> str:
        return f'P({category})'

    @property
    def _probability_to_value(self):
        return {
            self.prob_col('0'): 1,
            self.prob_col('1'): 2,
            self.prob_col('2+'): 3,
        }

    @classmethod
    def from_coefficients_csv(cls, csv_path: Path, *args, **kwargs):
        data = pd.read_csv(csv_path)

        if not set(['Feature', 'Coefficient']).issubset(set(data.columns)):
            raise ValueError('Function passed a CSV that does not appear to be model coefficients')

        data = data[['Feature', 'Coefficient']].fillna(0)

        data[['Segment', 'Category']] = data['Feature'].str.rsplit('_', n=1, expand=True)
        data['Category'] = data['Category'].fillna(0).astype(int)
        data.drop(columns=['Feature'], inplace=True)

        # Check to see if continuous parameters are also present
        continuous_params_path = csv_path.parent / 'scale_csv.csv'
        if continuous_params_path.is_file():
            continuous_params = pd.read_csv(continuous_params_path, index_col=0)
            continuous_params.columns = [c.rsplit('_', maxsplit=1)[0] for c in continuous_params.columns]
        else:
            continuous_params = None
        
        return cls(data, *args, **kwargs, continuous_params=continuous_params)

    @staticmethod
    def coefficients_to_combinations(coeff) -> pd.DataFrame:
        # To use `product`, we want tuples of each possible value by segment
        records_by_group = []

        for seg, segment_data in coeff.groupby('Segment'):
            # Rename columns to be unique per segment
            to_dump = segment_data.rename(columns={'Category': seg, 'Coefficient': f'Coeff_{seg}'}).drop(
                columns='Segment')

            # Dump to dictionary - we don't care about the index, so just take .values()
            # In other words, this will be something like [({seg: 1, Coeff_seg: 0.1234}, {seg: 2, Coeff_seg: 0.5678},)]
            records_by_group.append(to_dump.to_dict(orient='index').values())

        # A lot going on in this one line - product gives us possible segment value combos (e.g. [adults 1, children 1], [adults 1, children 2])
        # with their corresponding coefficients, reduce combines the dicts into one (we know they have unique keys from the above work),
        # and then convert to a DataFrame
        results_frame = pd.DataFrame(
            reduce(lambda left, right: {**left, **right}, combo) for combo in product(*records_by_group)
        )

        return results_frame[sorted(results_frame.columns, key=lambda c: c.replace('Coeff_', 'zzzzzz'))]

    @staticmethod
    def expand_to_zones(combinations, continuous_params, lookup_path) -> Tuple[str, pd.DataFrame]:
        lookup = pd.read_csv(lookup_path)

        # Figure out what the new zonal identifier should be
        zonal_column = [c for c in lookup.columns if
                        not (c.lower().startswith('cont_') or c.lower().startswith('cat_'))]
        assert len(zonal_column) == 1
        zonal_column = zonal_column[0]

        # Figure out which columns are categorical - we'll want to merge on these
        lookup.rename(columns={c: c.replace('CAT_', '') for c in lookup.columns}, inplace=True)
        zonal_attribute_columns = [c for c in lookup.columns if
                                   not (c.lower().startswith('cont_') or c == zonal_column)]

        # Merge into our combinations
        combined_dataset = combinations.merge(lookup, on=zonal_attribute_columns, how='right')

        # We always drop the intercept (category, not coefficient) column, and the zonal attribute flags (i.e. just used to expand to geography)
        columns_to_drop = ['intercept'] + zonal_attribute_columns

        if continuous_params is not None:
            # Set up the scaler according to the continuous params
            scaler = StandardScaler()
            scaler.mean_ = continuous_params.loc['mean'].values
            scaler.scale_ = continuous_params.loc['std'].values
            scaler.var_ = continuous_params.loc['var'].values

            continuous_cols = [f'CONT_{c}' for c in continuous_params.columns]

            # Apply the scaler to the "CONT_" columns (i.e. the _values_ rather than coefficients)
            combined_dataset[continuous_cols] = (
                scaler.transform(combined_dataset[continuous_cols])
            )

            # Now apply our continuous variables
            for col in continuous_cols:
                # Apply the uplift
                combined_dataset[col.replace('CONT_', 'Coeff_')] *= combined_dataset[col]
                # Drop both the uplifting column *and* the "category" column for that variable
                columns_to_drop += [col, col.replace('CONT_', '')]

        combined_dataset.drop(columns=columns_to_drop, inplace=True)

        combined_dataset.set_index([c for c in combined_dataset.columns if not c.startswith('Coeff_')], inplace=True)
        combined_dataset['Combined Coefficient'] = np.exp(-combined_dataset.sum(axis=1))

        return zonal_column, combined_dataset

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError(f'Can only multiply {type(self)} by another instance of the same (got {type(other)})')

        assert self.zonal_definition == other.zonal_definition

        joined = self.probabilities.join(other.probabilities)

        # Calculate dependent probabilities
        to_drop = []
        if self.dependent_category:
            assert self.dependent_category in other.probabilities.columns

            joined[self.case_category] *= joined[self.dependent_category]
            joined[self.noncase_category] *= joined[self.dependent_category]
            to_drop.append(self.dependent_category)

        if other.dependent_category:
            assert other.dependent_category in self.probabilities.columns

            joined[other.case_category] *= joined[other.dependent_category]
            joined[other.noncase_category] *= joined[other.dependent_category]
            to_drop.append(other.dependent_category)

        joined.drop(columns=to_drop, inplace=True)

        # Get into long form
        long_data = joined.reset_index().melt(id_vars=joined.index.names, var_name='car_availability',
                                              value_name='probability')

        long_data['car_availability'] = long_data['car_availability'].map(self._probability_to_value)

        return long_data.pivot_table(
            columns=self.zonal_definition, values='probability',
            index=[c for c in long_data.columns if c not in (self.zonal_definition, 'probability')]
        )


def _validate_segmentation(segs_to_check: list[str], dvector: DVector):
    """ Check segmentations are in a DVector. """
    for seg in segs_to_check:
        if not seg in dvector.segmentation.names:
            raise RuntimeError(f"{seg} is expected in your DVector but is not there.")


def _drop_values_and_total(
        dvector: DVector,
        vals_to_drop: list[int],
        segmentation: Optional[str] = 'car_availability'
) -> DVector:
    """ Drop segment values from a dvector, add 'total' segmentation and aggregate. """
    return dvector.drop_by_segment_values(
        segmentation, vals_to_drop
    ).add_segments(['total']).aggregate(['total'])


def _apply_norcom_shift(
        norcom_result: DVector,
        any_car_ownership_correction: float,
        multiple_car_ownership_correction: float,
        negative_infill_factor: float
) -> DVector:
    """Apply post-NorCOM adjustment to the 0v1+ and 1v2+ models.

    Parameters
    ----------
    norcom_result: DVector
        DVector with segmentation of 'car_availability'
    any_car_ownership_correction: float
        0v1+ Model Correction (applied relative to 0 cars category).
        -0.1 means that the 0 car owning households are reduced by
        10% and the 1+ car owning households are increased by 10%.
    multiple_car_ownership_correction: float
        1v2+ Model Correction (applied relative to the 1 car category).
        -0.2 means that the 1 car owning households are reduced by
        20% and the 2+ car owning households are increased by 20%.
    negative_infill_factor: float
        infill factor in case of negative households in a zone. This is applied
        as multiplicative to the initial number of households in a zone.
        An example is if a zone has 2 0 car owning households as predicted by
        the NorCOMResults, but then 10% of the 0v1+ model is removed from the
        no car owning category and put into the 1+ car owning category, then
        the zone would result in negative no car owning households and so that
        value is replaced with 2 * 0.5 = 1 household.

    Returns
    -------
    DVector
        Segmentation of 'car_availability'
    """

    # get model results
    # (0v1+)
    _0_cars = _drop_values_and_total(norcom_result,[2, 3])
    _1plus_cars = _drop_values_and_total(norcom_result,[1])
    # (1v2+)
    _1_car = _drop_values_and_total(norcom_result,[1, 3])
    _2plus_cars = _drop_values_and_total(norcom_result,[1, 2])

    # create DVectors of the adjustment values
    # TODO Cant currently multiply a DVector by a number (TypeError: unsupported operand type(s) for *: 'float' and 'DVector')
    _0_v_1_dvec = _0_cars.copy()
    _0_v_1_dvec.data.loc[:] = any_car_ownership_correction
    _1_v_2_dvec = _0_cars.copy()
    _1_v_2_dvec.data.loc[:] = multiple_car_ownership_correction

    # step 1: shift households between 0and1+ model, infill negatives with fudge
    _0_cars_step_1 = _0_cars + (_0_v_1_dvec * _1plus_cars)
    _0_cars_step_1.data = _0_cars_step_1.data.where(_0_cars_step_1.data > 0, negative_infill_factor * _0_cars.data)
    _1plus_cars_step_1 = _1plus_cars - (_0_v_1_dvec * _1plus_cars)

    # step 2: calculate new 1v2+ car numbers based on expected proportions of 1v2+ from the original model
    _1_car_step_2 = _1plus_cars_step_1 * (_1_car / (_1_car + _2plus_cars))
    _2plus_cars_step_2 = _1plus_cars_step_1 * (_2plus_cars / (_1_car + _2plus_cars))

    # step 3: shift households between 1and2+ model, infill negatives with fudge
    _1_car_step_3 = _1_car_step_2 + (_1_v_2_dvec * _2plus_cars_step_2)
    _1_car_step_3.data = _1_car_step_3.data.where(_1_car_step_3.data > 0, negative_infill_factor * _1_car_step_2.data)
    _2plus_cars_step_3 = _2plus_cars_step_2 - (_1_v_2_dvec * _2plus_cars_step_2)

    # combine output into a single dataframe
    # TODO ignore_index=True here means that the index will be 0,1,2 in the order in which they are concatenated, and therefore we can do index + 1 to get to car_availability seegmentation.
    output = pd.concat(
        [_0_cars_step_1.data, _1_car_step_3.data, _2plus_cars_step_3.data],
        ignore_index=True
    )
    output.index = output.index + 1
    output.index.name = 'car_availability'

    return create_dvector_from_data(
        dvector_data=output, geographical_level=norcom_result.zoning_system.name,
        input_segments=['car_availability']
    )


def apply_norcom(
        any_car_ownership_result: NorCOMResult,
        multiple_car_ownership_result: NorCOMResult,
        input_dvector: DVector,
        norcom_segmentations: Optional[list[str]] = None,
        any_car_ownership_correction: Optional[float] = -0.1,
        multiple_car_ownership_correction: Optional[float] = -0.2,
        negative_infill_factor: Optional[float] = 0.5
) -> DVector:
    """Apply NorCOM results to DVector.

    Parameters
    ----------
    any_car_ownership_result: NorCOMResult
        Result of the 0v1+ estimation model.
    multiple_car_ownership_result: NorCOMResult
        Result of the 1v2+ estimation model.
    input_dvector: DVector
        DVector to apply NorCOM to. Might be population or households, for example.
    norcom_segmentations: Optional[list[str]], default None
        The names of the segmentations in `input_dvector`, `any_car_ownership_result`, and
        `multiple_car_ownership_result` that mean the probabilities in the NorCOMResults
        can be multiplied with a DVector with the same or more segmentation.
    any_car_ownership_correction: Optional[float], default -0.1
        0v1+ Model Correction (applied relative to 0 cars category).
        -0.1 means that the 0 car owning households are reduced by
        10% and the 1+ car owning households are increased by 10%.
    multiple_car_ownership_correction: Optional[float], default -0.2
        1v2+ Model Correction (applied relative to the 1 car category).
        -0.2 means that the 1 car owning households are reduced by
        20% and the 2+ car owning households are increased by 20%.
    negative_infill_factor: Optional[float], default 0.5
        infill factor in case of negative households in a zone. This is applied
        as multiplicative to the initial number of households in a zone.
        An example is if a zone has 2 0 car owning households as predicted by
        the NorCOMResults, but then 10% of the 0v1+ model is removed from the
        no car owning category and put into the 1+ car owning category, then
        the zone would result in negative no car owning households and so that
        value is replaced with 2 * 0.5 = 1 household.

    Returns
    -------
    DVector
        As `input_dvector` with NorCOM applied, and therefore the `car_availability`
        segmentation is either added (if its not already in `input_dvector`) or
        replaced based on this calculation.
    """

    # this is because we don't want mutable args in the function def
    if norcom_segmentations is None:
        # these are the current default variables that we apply norcom to
        norcom_segmentations = ['accom_h', 'ns_sec', 'adults', 'children']

    # expand the results to have all three probability levels in one dataframe
    probability_result = any_car_ownership_result * multiple_car_ownership_result

    # convert to DVector with the right zone system subset
    probabilities = create_dvector_from_data(
        dvector_data=probability_result, geographical_level=input_dvector.zoning_system.name,
        input_segments=list(probability_result.index.names)
    )

    # initial segmentation of the input DVector
    initial_segmentation = input_dvector.segmentation.names
    # remove car availability if it's an existing dimension of the data
    if 'car_availability' in initial_segmentation:
        initial_segmentation.remove('car_availability')
        input_dvector = input_dvector.aggregate(initial_segmentation)

    # check norcom required segmentations are in input_dvector
    _validate_segmentation(segs_to_check=norcom_segmentations, dvector=input_dvector)

    # apply norcom to this modelled output
    apply_probabilities = input_dvector * probabilities

    # aggregate the post-norcom data to just car availability by zone
    result = apply_probabilities.aggregate(['car_availability'])

    # apply the global bias corrections
    adjusted_result = _apply_norcom_shift(
        norcom_result=result,
        any_car_ownership_correction=any_car_ownership_correction,
        multiple_car_ownership_correction=multiple_car_ownership_correction,
        negative_infill_factor=negative_infill_factor
    )

    # get back to full segmentation
    return apply_proportions(apply_probabilities, adjusted_result)
