from pathlib import Path
import pandas as pd

import land_use.preprocessing as pp
from land_use import constants, data_processing
from caf.base.zoning import TranslationWeighting
from caf.base import DVector

LMS_INPUT_DIR = Path(r'I:\NorMITs Land Use\2023\import\Labour Market and Skills\LMS_SIC_Ind2')
BASE_EMP_OUTPUTS_DIR = Path(r'F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs')

region_corr = pd.read_csv(Path(r'I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists',
                                  'GOR2021_CD_NM_EWS.csv'))
region_dict = dict(sorted(region_corr.values.tolist()))


def pre_process_lms_sic_for_control():
    """
    Function to read in and pre-process the Labour Market & Skills dataset for SIC Industry Table 2
    Outputs growth factors for each forecast year
    """
    sic = []
    # Read in and format the LM&S data for each region
    for region in region_corr['RGN21NM']:
        df = pd.read_csv(LMS_INPUT_DIR / fr'LMS_Ind2_{region}.csv',
                         header=[0], skiprows=[1]).dropna()
        df = df.rename(columns={df.columns[0]: 'Industry'})
        df['region'] = region
        sic.append(df)
    sic_rgns = pd.concat(sic)

    # Map LM&S industries to our segmentation
    lms_sic_corr = pd.read_csv(LMS_INPUT_DIR / r'LMS_SIC_1_digit_corr.csv',
                               dtype={'LU_SIC_1_digit': int})
    sic_rgns = pd.merge(sic_rgns, lms_sic_corr,
                        left_on='Industry', right_on='Labour Market & Skills', how='left').dropna()
    # TODO question, 2025 to 2035 or 2015 to 2035?
    sic_rgns = sic_rgns[['LU_SIC_1_digit', 'region', '2025', '2035']].groupby(
        by=['LU_SIC_1_digit', 'region'], as_index=False)[['2025', '2035']].sum()

    # Get jobs into 1000s and aggregate any SIC values with multiple LM&S definitions, e.g. SIC 3
    sic_rgns['2025'] = sic_rgns['2025'] * 1000
    sic_rgns['2035'] = sic_rgns['2035'] * 1000
    # TODO numbers under 10,000?

    # Calculate % growth for 1 year
    sic_rgns['%_growth_1yr'] = ((sic_rgns['2035'] - sic_rgns['2025']) / sic_rgns['2025']) / 10

    # Using the Land Use base employment by SIC 1 digit,
    # apply these % growths for the various forecast years
    output_e6_dv = DVector.load(BASE_EMP_OUTPUTS_DIR / 'Output E6.hdf')
    # Translate output into Region zoning
    output_e6_dv = output_e6_dv.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False
    )
    # Aggregate segmentation to sic_1_digit
    output_e6_dv = output_e6_dv.aggregate(['sic_1_digit'])
    output_e6_df = output_e6_dv.data.rename(columns=region_dict).reset_index()
    output_e6_df = pd.melt(output_e6_df, id_vars=['sic_1_digit'], var_name='region')
    # Merge the growths from sic_rgns onto the Land Use E6 output (fill SIC -1, 20, 21 with 0s)
    control_totals = pd.merge(output_e6_df, sic_rgns, left_on=['sic_1_digit', 'region'],
                              right_on=['LU_SIC_1_digit', 'region'], how='left').fillna(0)

    # Apply the % growth to the Base Land Use totals and output as hdf,
    # ready to be read in as DVector
    f_years = {'2028': 5, '2033': 10, '2038': 15, '2043': 20, '2048': 25}
    for year in f_years.keys():
        f_year_control = control_totals.copy()
        f_year_control['control'] = (1 + f_year_control['%_growth_1yr'] * f_years.get(year))
        f_year_control = (f_year_control[['sic_1_digit', 'region', 'control']].
                          rename(columns={'control': year}))
        # Remap region back to codes
        f_year_control['region'] = f_year_control['region'].map(
            dict((x, y) for y, x in region_dict.items()))

        # Into a wide format for DVector
        f_year_control = pp.pivot_to_dvector(
            data=f_year_control,
            zoning_column='region',
            index_cols=['sic_1_digit'],
            value_column=year,
        )
        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR / 'LMS_SIC_Ind2.hdf',
            df=f_year_control,
            multiple_output_ref=year
        )
