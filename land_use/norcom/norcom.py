from functools import reduce
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
