
import os

import pandas as pd

import land_use.lu_constants as consts
import land_use.utils.compress as comp
import land_use.utils.general as utils
import land_use.utils.file_ops as fo
import land_use.utils.normalise_tts as norm
from land_use.future_land_use_DDG import NTEM_fy_process


class FutureYearLandUse:
    def __init__(self,
                 model_folder=consts.LU_FOLDER,
                 iteration=consts.FYLU_MR_ITER,
                 import_folder=consts.LU_IMPORTS,
                 by_folder=consts.BY_FOLDER,
                 fy_folder=consts.FY_FOLDER,
                 model_zoning='msoa',
                 zones_folder=consts.ZONES_FOLDER,
                 base_resi_land_use_path=None,
                 base_non_resi_land_use_path=None,
                 area_type_path=consts.LU_AREA_TYPES,
                 ctripend_database_path=consts.CTripEnd_Database,
                 fy_demographic_path=None,
                 fy_at_mix_path=None,
                 fy_soc_mix_path=None,
                 base_year='2018',
                 future_year=None,
                 scenario_name=None,
                 CAS_scen=None,
                 pop_growth_path=None,
                 emp_growth_path=None,
                 ca_growth_path=None,
                 ca_shares_path=None,
                 pop_segmentation_cols=None,
                 sub_for_defaults=False):

        # TODO: Add versioning

        # File ops
        self.model_folder = model_folder
        self.iteration = iteration
        self.import_folder = model_folder + '/' + import_folder
        self.zones_folder = zones_folder
        self.by_folder = by_folder
        self.fy_folder = fy_folder
        self.by_home_folder = model_folder + '/' + by_folder + '/' + iteration
        self.fy_home_folder = model_folder + '/' + fy_folder + '/' + iteration


        # Basic config
        self.model_zoning = model_zoning
        self.base_year = base_year
        self.future_year = future_year
        self.scenario_name = scenario_name
        self.CAS_scen = CAS_scen
        self.area_type_path = area_type_path
        self.CTripEnd_Database_path = ctripend_database_path

        # If Nones passed in, set defaults
        # This is for base datasets that don't vary between scenarios
        if base_resi_land_use_path is None:
            if sub_for_defaults:
                print('Using default Residential Land Use')
                base_resi_land_use_path = consts.RESI_LAND_USE_MSOA
                print(base_resi_land_use_path)
            else:
                raise ValueError('No base land use provided')
        if base_non_resi_land_use_path is None:
            if sub_for_defaults:
                print('Using default non-Residential Land Use')
                base_non_resi_land_use_path = consts.RESI_LAND_USE_MSOA
                print(base_resi_land_use_path)

        # If Nones passed in, parse paths
        # This is for datasets that change between scenarios
        if fy_demographic_path is None:
            try:
                fy_demographic_path = self._get_scenario_path('dem_mix')
            except:
                if sub_for_defaults:
                    print('Using default demographic mix')
                    fy_demographic_path = consts.NTEM_DEMOGRAPHICS_MSOA
                else:
                    print('Demographic mix init failed')
        if fy_at_mix_path is None:
            try:
                fy_at_mix_path = self._get_scenario_path('at_mix')
            except:
                print('Future year area type mix init failed')
                fy_at_mix_path = None
        if fy_soc_mix_path is None:
            try:
                fy_soc_mix_path = self._get_scenario_path('soc_mix')
            except:
                print('Future year soc mix init failed')
                fy_soc_mix_path = None
        if pop_growth_path is None:
            pop_growth_path = self._get_scenario_path('pop_growth')
        if emp_growth_path is None:
            emp_growth_path = self._get_scenario_path('emp_growth')
        if ca_growth_path is None:
            try:
                ca_growth_path = self._get_scenario_path('ca_growth')
            except:
                print('CA growth init failed')
                ca_growth_path = None
        if ca_shares_path is None:
            try:
                ca_shares_path = self._get_scenario_path('ca_shares')
            except:
                print('CA shares init failed')
                ca_shares_path = None

        # Segmentation
        self.pop_segmentation_cols = pop_segmentation_cols

        # Build paths
        write_folder = os.path.join(
            model_folder,
            fy_folder,
            iteration)
        # write_folder = os.path.join(
        #     model_folder,
        #     consts.FY_FOLDER,
        #     iteration,
        #     'outputs',
        #     'scenarios',
        #     scenario_name)

        pop_write_name = os.path.join(
            write_folder,
            ('land_use_' + str(self.future_year) + '_pop.csv'))

        emp_write_name = os.path.join(
            write_folder,
            ('land_use_' + str(self.future_year) + '_emp.csv'))

        report_folder = os.path.join(write_folder,
                                     'reports')

        # Build folders
        if not os.path.exists(write_folder):
            fo.create_folder(write_folder)
        if not os.path.exists(report_folder):
            fo.create_folder(report_folder)

        # Set object paths
        self.in_paths = {
            'iteration': iteration,
            'model_zoning': model_zoning,
            'base_year': base_year,
            'future_year': future_year,
            'scenario_name': scenario_name,
            'base_resi_land_use': base_resi_land_use_path,
            'base_non_resi_land_use': base_non_resi_land_use_path,
            'fy_dem_mix': fy_demographic_path,
            'fy_at_mix': fy_at_mix_path,
            'fy_soc_mix': fy_soc_mix_path,
            'pop_growth': pop_growth_path,
            'emp_growth': emp_growth_path,
            'ca_growth': ca_growth_path,
            'ca_shares': ca_shares_path
            }

        self.out_paths = {
            'write_folder': write_folder,
            'report_folder': report_folder,
            'pop_write_path': pop_write_name,
            'emp_write_path': emp_write_name
        }

        # Write init reports for param audits
        init_report = pd.DataFrame(self.in_paths.values(),
                                   self.in_paths.keys()
                                   )
        init_report.to_csv(
            os.path.join(self.out_paths['report_folder'],
                         '%s_%s_input_params.csv' % (self.scenario_name,
                                                   self.future_year))
        )

    def NTEM_pop(self):
        NTEM_fy_process.ntem_fy_pop_interpolation(self)
    def build_fy_pop(self,
                     balance_demographics=True,
                     adjust_ca=True,
                     adjust_soc=True,
                     adjust_area_type=True,
                     ca_growth_method='factor',
                     normalise=True,
                     export=True,
                     verbose=True,
                     reports=True):
        """
        """
        # TODO: Replace with multi-dimensional control

        # Build population
        fy_pop, pop_reports = self._grow_pop(verbose=verbose)

        if balance_demographics:
            fy_pop, dem_reports = self._balance_demographics(fy_pop,
                                                             reports=reports)
        else:
            dem_reports = dict({'null_report': 0})

        # Adjust car availability mix
        if adjust_ca:
            fy_pop, ca_changes = self._adjust_ca(fy_pop,
                                                 ca_growth_method=ca_growth_method)
            if reports:
                ca_changes.to_csv(
                    os.path.join(
                        self.out_paths['report_folder'],
                        'ca_changes_%s.csv' % str(self.future_year)),
                    index=False
                )

        # TODO: Adjust SOC
        if adjust_soc:
            fy_pop = self._adjust_soc(fy_pop)
            
        if adjust_area_type:
            # Adjust area type
            fy_pop, at_changes = self._adjust_area_type(fy_pop)
            print('Adjusting area type')
            if reports:
                at_changes.to_csv(
                    os.path.join(self.out_paths['report_folder'],
                                 'area_type_changes_%s.csv' % str(self.future_year)),
                    index=False
                )

        # Reporting
        if reports:
            pop_reports[0].to_csv(
                os.path.join(self.out_paths['report_folder'],
                             'pop_changes_%s_for_%s.csv' % (str(self.base_year),
                                                            str(self.future_year))),
                index=False)

            pop_reports[1].to_csv(
                os.path.join(self.out_paths['report_folder'],
                             'pop_changes_%s.csv' % str(self.future_year)),
                index=False
            )

            for dr in list(dem_reports):
                dem_reports[dr].to_csv(
                    os.path.join(self.out_paths['report_folder'],
                                 'dem_changes_%s_%s.csv' % (str(dr),
                                                            str(self.future_year))),
                    index=False
                )

        # Normalise to tfn tt
        if normalise:
            fy_pop = norm.expanded_to_normalised(fy_pop,
                                                 var_col=self.future_year)

        # Export main dataset
        if export:
            if verbose:
                print('Writing to:')
                print(self.out_paths['pop_write_path'])
            comp.write_out(fy_pop, self.out_paths['pop_write_path'])

        return fy_pop

    def build_fy_emp(self,
                     export=True,
                     verbose=True):

        fy_emp = self._grow_emp(verbose=verbose)

        if export:
            if verbose:
                print('Writing to:')
                print(self.out_paths['emp_write_path'])
            comp.write_out(fy_emp, self.out_paths['emp_write_path'])

        return fy_emp

    def _get_scenario_path(self,
                           vector):
        """
        Function to build a path to the relevant scenario input.
        This can scale indefinitely as long as it's private.

        Parameters
        ----------
        vector = one from ['dem_mix', 'at_mix, 'soc_mix', 'pop_growth',
        'emp_growth', 'ca_shares', 'ca_growth']

        Returns
        -------
        path : path to required vector
        """
        if vector == 'dem_mix':
            target_folder = 'demographics'
            target_file = 'future_demographic_values.csv'
        elif vector == 'at_mix':
            target_folder = 'area types'
            target_file = 'future_area_type_mix.csv'
        elif vector == 'soc_mix':
            target_folder = 'soc_mix'  # Deprecated??
            target_file = 'future_soc_mix.csv'
        elif vector == 'pop_growth':
            target_folder = 'population'
            target_file = 'future_population_growth.csv'
        elif vector == 'emp_growth':
            target_folder = 'employment'
            target_file = 'future_employment_growth.csv'
        elif vector == 'ca_shares':
            target_folder = 'car ownership'
            target_file = 'ca_future_shares.csv'
        elif vector == 'ca_growth':
            target_folder = 'car ownership'
            target_file = 'ca_future_growth.csv'
        else:
            raise ValueError('Not sure where to look for ' + vector)

        # Run scenario name through consts to get name
        sc_path = os.path.join(
            self.model_folder,
            self.import_folder,
            'scenarios',
            consts.SCENARIO_FOLDERS[self.scenario_name],
            target_folder,
            target_file
        )

        return sc_path

    def _adjust_area_type(self,
                          fy_pop: pd.DataFrame,
                          verbose=False):

        """
        Parameters
        ----------
        fy_pop: pd.Dataframe
        Data frame of population for future year by land use zoning (MSOA)

        Returns
        -------
        fy_pop_w_at = Dataframe with future adjusted area types

        changes = List of area types that have changed, for reporting

        """
        # Define model zone
        model_zone = self.model_zoning.lower() + '_zone_id'

        # Import and process future year area type
        fy_at = pd.read_csv(self.in_paths['fy_at_mix'])
        fy_at = fy_at.reindex([model_zone,
                               self.future_year], axis=1)
        fy_at = fy_at.rename(columns={self.future_year: 'fy_at'})

        # Merge future onto base
        fy_pop_w_at = fy_pop.merge(fy_at,
                                   how='left',
                                   on=model_zone)

        # Make report
        changes = fy_pop_w_at[
            fy_pop_w_at['area_type'] != fy_pop_w_at['fy_at']]
        changes = changes.reindex(
            [model_zone, 'area_type', 'fy_at'], axis=1).drop_duplicates()
        changes = changes.reset_index(drop=True)

        # Re pick columns and drop
        fy_pop_w_at['area_type'] = fy_pop_w_at['fy_at']
        fy_pop_w_at = fy_pop_w_at.drop('fy_at', axis=1)

        return fy_pop_w_at, changes

    def _grow_pop(self,
                  verbose=False
                  ):

        # Define zone col name
        zone_col = self._define_zone_col()

        #TODO: Check if bases are misaligned

        # Get pop growth, filter to target year only
        population_growth = self._get_fy_pop_emp(
            'pop',
            retain_cols=['soc', 'ns'])  # If there's NPR segments, keep them

        # ## BASE YEAR POPULATION ## #
        print("Loading the base year population data...")
        base_year_pop = utils.get_land_use(
            self.in_paths['base_resi_land_use'],
            model_zone_col=zone_col,
            segmentation_cols=None)
        base_year_pop = base_year_pop.rename(
            columns={'people': self.base_year})
        by_pop_report = utils.lu_out_report(base_year_pop,
                                            pop_var=self.base_year)

        # TODO: Add the traveller type join back on
        # TODO: use the normalise tts script
        if 'tfn_traveller_type' in list(base_year_pop):
            base_year_pop = utils.infill_traveller_types(
                land_use_build=base_year_pop,
                traveller_type_lookup=consts.TFN_TT_INDEX,
                attribute_subset=None,
                left_tt_col='tfn_traveller_type',
                right_tt_col='tfn_traveller_type')

        # Audit population numbers
        print("Base Year Population: %d" % base_year_pop[self.base_year].sum())

        # ## FUTURE YEAR POPULATION ## #
        print("Generating future year population data...")
        # Merge on all possible segmentations - not years
        merge_cols = utils.intersection(list(base_year_pop),
                                        list(population_growth))

        # If merge cols is only msoa, it's a tt dataset, apply the index
        if len(merge_cols) == 1 or 'tfn_travelller_type' in merge_cols:
            base_year_pop = norm.normalised_to_expanded(base_year_pop,
                                                        drop_tt=True)
            merge_cols = utils.intersection(list(base_year_pop),
                                            list(population_growth))

        # TODO: Check that I don't need this anymore
        # Control data types
        base_year_pop['soc'] = base_year_pop['soc'].astype(float).astype(int)
        base_year_pop['ns'] = base_year_pop['ns'].astype(float).astype(int)
        population_growth['soc'] = population_growth['soc'].astype(float).astype(int)
        population_growth['ns'] = population_growth['ns'].astype(float).astype(int)

        population = self._grow_to_future_year(
            by_vector=base_year_pop,
            fy_vector=population_growth,
            merge_cols=merge_cols
        )
        fy_pop_report = utils.lu_out_report(population,
                                            pop_var=self.future_year)

        # Population Audit
        if verbose:
            print('Merged on:')
            print(merge_cols)
            print(list(population))
            print('\n', '-' * 15, 'Population Audit', '-' * 15)
            print('Total population for year %s is: %.4f' % (self.future_year, population[self.future_year].sum()))
            print('\n')

        # Write the produced population to file
        # print("Writing population to file...")
        # population_output = os.path.join(out_path, self.pop_fname)
        # population.to_csv(population_output, index=False)
        report = [by_pop_report, fy_pop_report]

        return population, report

    def _grow_emp(self,
                  verbose=False):
        # Init
        zone_col = self._define_zone_col()
        emp_cat_col = 'employment_cat'

        employment_growth = self._get_fy_pop_emp(
            'emp',
            retain_cols=['soc', 'ns'])

        # Check for soc 4, infill if not there
        employment_growth = self._add_soc_4(employment_growth,
                                            soc_col='soc')

        # ## BASE YEAR EMPLOYMENT ## #
        print("Loading the base year employment data...")
        base_year_emp = utils.get_land_use(
            path=self.in_paths['base_non_resi_land_use'],
            model_zone_col=zone_col,
            segmentation_cols=None,
            add_total=False,
            to_long=False)

        # Used to have consistent future year col for var, but now may be people
        # This will stop the growth working
        if 'people' in base_year_emp:
            base_year_emp = base_year_emp.rename(
                columns={'people': self.base_year})

        # Print employment numbers
        print('Base year employment total %d' % base_year_emp[self.base_year].sum())
        print('Includes non-working segments \n \n')

        # ## FUTURE YEAR EMPLOYMENT ## #
        print('Generating future year employment data...')

        # Merge on all possible segmentations - not years
        merge_cols = utils.intersection(list(base_year_emp), list(employment_growth))

        employment = self._grow_to_future_year(
            by_vector=base_year_emp,
            fy_vector=employment_growth,
            merge_cols=merge_cols)

        employment = employment.rename(columns={self.base_year: 'people'})

        print(list(employment))

        return employment

    def _adjust_ca(self,
                   fy_pop_vector: pd.DataFrame,
                   ca_growth_method: str,
                   verbose=True) -> pd.DataFrame:
        """

        Parameters
        ----------
        fy_pop_vector: Ready adjusted future year pop vector
        ca_growth_method: flat adjustment in fy or factor based growth

        Returns
        -------
        fy_pop_vector: Population vector with adjusted car availability
        ca_adjustment_factors: Report Dataframe

        """
        # Get zone name
        zone_col = self._define_zone_col()

        # Build base year ca totals
        # This doesn't touch the future year vector, yet
        by_ca = fy_pop_vector.copy()
        print(list(by_ca))
        # TODO: Check ca in cols

        by_ca = by_ca.reindex(
            [zone_col, 'ca', self.future_year], axis=1).groupby(
            [zone_col, 'ca']).sum().reset_index()

        # Get the relative shares of each segment
        by_ca_share = by_ca.copy()
        by_ca_share = by_ca_share.pivot(
            index=zone_col,
            columns='ca',
            values=self.future_year).reset_index()

        by_ca_share['total'] = by_ca_share[1] + by_ca_share[2]
        by_ca_share[1] /= by_ca_share['total']
        by_ca_share[2] /= by_ca_share['total']
        by_ca_share = by_ca_share.drop('total', axis=1)
        by_ca_share = by_ca_share.melt(id_vars=zone_col,
                                       var_name='ca',
                                       value_name=self.future_year)
        by_ca_share = by_ca_share.rename(
            columns={self.future_year: 'by_ca'})
        by_ca_share = by_ca_share.sort_values([zone_col, 'ca']).reset_index(drop=True)

        # So:
        # base year ca totals are in by_ca
        # each segment over sum is in by_ca_share

        if ca_growth_method == 'flat':
            # Get ca shares
            # TODO: Import growth or flat depending on method
            ca_shares = pd.read_csv(self.in_paths['ca_shares'])
            # Filter to target_year
            ca_shares = ca_shares.reindex(
                [zone_col, 'ca', self.future_year], axis=1)
            ca_shares = ca_shares.rename(columns={self.future_year: 'fy_ca'})

            # Join
            fy_ca_factors = by_ca_share.merge(
                ca_shares,
                how='left',
                on=[zone_col, 'ca'])
            fy_ca_factors['ca_adj'] = fy_ca_factors['fy_ca'] / fy_ca_factors['by_ca']
            fy_ca_factors = fy_ca_factors.drop(['by_ca', 'fy_ca'], axis=1)

        elif ca_growth_method == 'factor':
            # Get ca growth

            # adj factor = ((fy growth factor * base year share) / Sum(fy
            # growth factor * base year share)) / base year share fy
            # ca = base year demand * adj factor

            ca_factors = pd.read_csv(self.in_paths['ca_growth'])
            # Filter to target year
            ca_factors = ca_factors.reindex(
                [zone_col, 'ca', self.future_year], axis=1)
            ca_factors = ca_factors.rename(columns={self.future_year: 'fy_ca'})

            # Join
            ca_shares = by_ca_share.merge(
                ca_factors,
                how='left',
                on=[zone_col, 'ca'])
            # Sum total so control factors to 1 (itself)
            ca_shares['fy_ca'] *= ca_shares['by_ca']

            totals = ca_shares.groupby('msoa_zone_id')['fy_ca'].sum().reset_index()
            totals = totals.rename(columns={'fy_ca': 'fy_ca_tot'})

            fy_ca_factors = ca_shares.merge(
                totals,
                how='left',
                on=zone_col
            )
            fy_ca_factors['fy_ca'] /= fy_ca_factors['fy_ca_tot']
            fy_ca_factors['ca_adj'] = fy_ca_factors['fy_ca']/fy_ca_factors['by_ca']

            fy_ca_factors = fy_ca_factors.drop(
                ['by_ca', 'fy_ca', 'fy_ca_tot'], axis=1)

        before = fy_pop_vector[self.future_year].sum()
        ca_before = fy_pop_vector.groupby(['ca'])[self.future_year].sum()

        # Adjust CA
        fy_pop_vector = fy_pop_vector.merge(
            fy_ca_factors,
            how='left',
            on=[zone_col, 'ca'])
        fy_pop_vector[self.future_year] *= fy_pop_vector['ca_adj']
        fy_pop_vector = fy_pop_vector.drop('ca_adj', axis=1)

        after = fy_pop_vector[self.future_year].sum()
        ca_after = fy_pop_vector.groupby(['ca'])[self.future_year].sum()

        ca_changes = pd.DataFrame(ca_after)

        if verbose:
            print('*' * 15)
            print('Car availability adjustment')
            print('CA growth method: %s' % ca_growth_method)
            print('Total before: ' + str(before.astype(int)))
            print('Total after: ' + str(after.astype(int)))
            print('Shares before: ' + str(ca_before.astype(int)))
            print('Shares after: ' + str(ca_after.astype(int)))

        return fy_pop_vector, ca_changes

    def _balance_demographics(self,
                              fy_pop,
                              reports=False,
                              verbose=True):
        """
        Reshape future year population to match a given demographic vector

        Parameters
        ----------
        fy_pop:
            Vector of future year population, already grown in volume

        Returns
        -------
        fy_pop:
            FY pop with age segments adjusted to demographic outputs
        segment_change_report:
            Report detailing how segments have fared with the change
        """
        # Get totals by segment before and after
        intact_cols = list(fy_pop)

        if verbose:
            print('Pop before = %d' % fy_pop[self.future_year].sum())

        # Get target demographic mix data
        demographics = pd.read_csv(self.in_paths['fy_dem_mix'])
        # Get demographic factors
        target_factors = self._get_age_factors(demographics)
        target_factors = target_factors.rename(
            columns={'factor': 'target_factor'})

        # Infill traveller types
        fy_pop = norm.infill_ntem_tt(fy_pop)

        # Report on all tt segments
        report_cols = list(fy_pop)
        report_cols.remove('msoa_zone_id')
        report_cols.remove(self.future_year)
        if reports:
            report_dict = dict()
            for rc in report_cols:
                before_report = utils.lu_out_report(
                    fy_pop,
                    pop_var=self.future_year,
                    group_vars=[rc])
                before_report = before_report.rename(
                    columns={self.future_year: 'before'}
                )
                report_dict.update({rc: before_report})

        # Get current factors
        current_factors = self._get_age_factors(fy_pop)
        current_factors = current_factors.rename(
            columns={'factor': 'current_factor'})

        # Get correction factors
        corr_factors = current_factors.merge(
            target_factors, how='left', on=['msoa_zone_id', 'age']
        )
        corr_factors['corr'] = corr_factors['target_factor'] / corr_factors['current_factor']

        # Apply correction factors
        fy_pop = fy_pop.merge(corr_factors,
                              how='left',
                              on=['msoa_zone_id', 'age'])
        fy_pop[self.future_year] *= fy_pop['corr']

        if reports:
            for rc in report_cols:
                after_report = utils.lu_out_report(
                    fy_pop,
                    pop_var=self.future_year,
                    group_vars=[rc])
                after_report = after_report.rename(
                    columns={self.future_year: 'after_adj'})
                # Merge
                after_report = report_dict[rc].merge(
                    after_report,
                    on=rc)
                report_dict.update({rc: after_report})

        else:
            report_dict = dict()

        # Back to original cols
        fy_pop = fy_pop.reindex(intact_cols, axis=1)

        if verbose:
            print('Pop after = %d' % fy_pop[self.future_year].sum())
            if reports:
                print('Adjustment before/after')
                print(report_dict['age'])

        return fy_pop, report_dict

    def _grow_to_future_year(self,
                             by_vector: pd.DataFrame,
                             fy_vector: pd.DataFrame,
                             merge_cols=None,
                             verbose=True) -> pd.DataFrame:
        """
        Parameters
        ----------
        by_vector: Dataframe of a base year pop/emp vector
        fy_vector: Dataframe of growth factors from a given base year
        to a give future year, by year

        Returns
        -------
        fy_vector: Absolute totals from by_vector, grown to fy totals
        """
        if merge_cols is None:
            merge_cols = self._define_zone_col()
        ####
        print(merge_cols)
        print(by_vector[merge_cols])
        print(fy_vector[merge_cols])
        ###

        start = by_vector[self.base_year].sum()

        fy_vector = by_vector.merge(fy_vector,
                                    how='left',
                                    on=merge_cols)
        merge = fy_vector[self.base_year].sum()

        fy_vector[self.future_year] *= fy_vector[self.base_year]
        end = fy_vector[self.future_year].sum()
        fy_vector = fy_vector.drop(self.base_year, axis=1)

        if verbose:
            print('Start: ' + str(start))
            print('Merge:' + str(merge))
            print('End:' + str(end))

        return fy_vector

    def _get_soc_weights(self,
                         zone_col: str = 'msoa_zone_id',
                         soc_col: str = 'soc_class',
                         jobs_col: str = 'seg_jobs',
                         str_cols: bool = False
                         ) -> pd.DataFrame:
        """
        Converts the input file into soc weights by zone

        Parameters
        ----------
        zone_col:
            The column name in soc_weights_path that contains the zone data.

        soc_col:
            The column name in soc_weights_path that contains the soc categories.

        jobs_col:
            The column name in soc_weights_path that contains the number of jobs
            data.

        str_cols:
            Whether the return dataframe columns should be as [soc1, soc2, ...]
            (if True), or [1, 2, ...] (if False).

        Returns
        -------
        soc_weights:
            a wide dataframe with zones from zone_col as the column names, and
            soc categories from soc_col as columns. Each row of soc weights will
            sum to 1.
        """
        # Init
        soc_weighted_jobs = pd.read_csv(self.in_paths['base_soc_mix'])

        # Convert soc numbers to names (to differentiate from ns)
        soc_weighted_jobs[soc_col] = soc_weighted_jobs[soc_col].astype(int).astype(str)

        if str_cols:
            soc_weighted_jobs[soc_col] = 'soc' + soc_weighted_jobs[soc_col]

        # Calculate Zonal weights for socs
        # This give us the benefit of model purposes in HSL data
        group_cols = [zone_col, soc_col]
        index_cols = group_cols.copy()
        index_cols.append(jobs_col)

        soc_weights = soc_weighted_jobs.reindex(index_cols, axis='columns')
        soc_weights = soc_weights.groupby(group_cols).sum().reset_index()
        soc_weights = soc_weights.pivot(
            index=zone_col,
            columns=soc_col,
            values=jobs_col
        ).reset_index()

        # Convert to factors
        soc_segments = soc_weighted_jobs[soc_col].unique()
        soc_weights['total'] = soc_weights[soc_segments].sum(axis='columns')

        for soc in soc_segments:
            soc_weights[soc] /= soc_weights['total']

        soc_weights = soc_weights.drop('total', axis='columns')

        return soc_weights

    @staticmethod
    def _split_by_soc(df: pd.DataFrame,
                      soc_weights: pd.DataFrame,
                      zone_col: str = 'msoa_zone_id',
                      p_col: str = 'p',
                      unique_col: str = 'trips',
                      soc_col: str = 'soc',
                      split_cols: str = None
                      ) -> pd.DataFrame:
        """
        Splits df purposes by the soc_weights given.

        Parameters
        ----------
        df:
            Dataframe to add soc splits too. Must contain the following columns
            [zone_col, p_col, unique_col]

        soc_weights:
            Wide dataframe containing the soc splitting weights. Must have a
            zone_col columns, and all other columns are the soc categories to split
            by.

        zone_col:
            The name of the column in df and soc_weights that contains the
            zone data.

        p_col:
            Name of the column in df that contains purpose data.

        unique_col:
            Name of the column in df that contains the unique data (usually the
            number of trips at that row of segmentation)

        soc_col:
            The name to give to the added soc column in the return dataframe.

        split_cols:
            Which columns are being split by soc. If left as None, only zone_col
            is used.

        Returns
        -------
        soc_split_df:
            df with an added soc_col. Unique_col will be split by the weights
            given
        """
        # Init
        soc_cats = list(soc_weights.columns)
        # Drop zone col if it's made its way in
        soc_cats = [x for x in soc_cats if zone_col not in x]
        split_cols = [zone_col] if split_cols is None else split_cols

        # Figure out which rows need splitting
        if p_col in df:
            mask = (df[p_col].isin(consts.SOC_P))
            split_df = df[mask].copy()
            retain_df = df[~mask].copy()
            id_cols = split_cols + [p_col]
        else:
            # Split on all data
            split_df = df.copy()
            retain_df = None
            id_cols = split_cols

        # Split by soc weights
        split_df = pd.merge(
            split_df,
            soc_weights,
            on=zone_col
        )

        for soc in soc_cats:
            split_df[soc] *= split_df[unique_col]

        # Tidy up the split dataframe ready to re-merge
        split_df = split_df.drop(unique_col, axis='columns')
        # Re melt - get soc back as col
        split_df = split_df.melt(
            id_vars=id_cols,
            value_vars=soc_cats,
            var_name=soc_col,
            value_name=unique_col,
        )

        # Don't need to stick back together
        if retain_df is None:
            return split_df

        # Add the soc col to the retained values to match
        retain_df[soc_col] = 0

        # Finally, stick the two back together
        return pd.concat([split_df, retain_df])

    def _get_fy_pop_emp(self,
                        vector_type,
                        retain_cols):
        """
        vector_type = 'pop' or 'emp'
        """

        if vector_type == 'pop':
            dat = pd.read_csv(self.in_paths['pop_growth'])
        elif vector_type == 'emp':
            dat = pd.read_csv(self.in_paths['emp_growth'], dtype={'soc': str})
        ri_cols = list([self.model_zoning + '_zone_id'])
        for col in retain_cols:
            if col in list(dat):
                ri_cols.append(col)
        ri_cols.append(self.future_year)
        dat = dat.reindex(ri_cols, axis=1)

        return dat

    def _define_zone_col(self):
        """
        Work out a sensible column name to use as the model zoning id.

        Returns
        -------
        mz_id: A model zoning ID

        """
        if 'zone_id' not in self.model_zoning:
            zone_col = self.model_zoning.lower() + '_zone_id'
        else:
            zone_col = self.model_zoning.lower()

        return zone_col


    def _adjust_soc(self,
                    fy_pop):
        """
        fy_pop:
        future year population vector

        Returns
        -------
        soc_adjusted_pop:


        fy_soc:
        Summary report of
        """
        # Import SY SIC mix by scenario
        # Translate SIC mix to SOC
        # Translate FY SOC mix to % share by zone

        # (Export FY SOC mix by zone)
        # Factor FY pop to match zone mix

        # Audit soc mix against LA level from NELUM
        return 0    # soc_adjusted_pop, fy_soc

    def _get_age_factors(self,
                         age_vector: pd.DataFrame,
                         pop_col: str = None):
        """
        Get age factors from a df with age in

        Parameters
        ----------
        self
        age_vector

        Returns
        -------

        """

        if pop_col is None:
            pop_col = self.future_year

        # Reindex
        """
        One liner challenge
        
        """

        dem_cols = ['msoa_zone_id', 'age', pop_col]
        age_vector = age_vector.reindex(dem_cols, axis=1)

        # Get total
        age_vector['factor'] = age_vector[pop_col] / age_vector.groupby(
            'msoa_zone_id')[pop_col].transform('sum')

        ri_cols = ['msoa_zone_id', 'age', 'factor']
        rig_cols = ri_cols.copy()
        rig_cols.remove('factor')
        age_factors = age_vector.reindex(ri_cols, axis=1).\
            groupby(rig_cols).sum().reset_index()

        return age_factors

    def _add_soc_4(self,
                   employment_growth: pd.DataFrame,
                   soc_col: str = 'soc'):
        """
        Takes a set of growth vectors and adds soc 4, if needed
        """

        # Check for soc 4
        if '4' not in employment_growth[soc_col].drop_duplicates().to_list():
            _already_in = False

        out_growth = employment_growth.copy()
        prior_emp_growth = employment_growth.copy()

        if not _already_in:
            # Build mean factor for unm
            soc_4_growth = employment_growth.copy()
            soc_4_growth = soc_4_growth.groupby(
                [self.model_zoning + '_zone_id'])[self.future_year].mean()
            soc_4_growth = soc_4_growth.reset_index()
            soc_4_growth[soc_col] = '4'

            out_growth = pd.concat([prior_emp_growth, soc_4_growth])
            out_growth = out_growth.sort_values(
                [self.model_zoning + '_zone_id', soc_col])

        return out_growth
