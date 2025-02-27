# %%
from pathlib import Path

import pandas as pd

import land_use.preprocessing as pp


FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048]
POPULATION_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs")
HOUSEHOLDS_DIR = Path(r'I:\NorMITs Land Use\2023\import\ONS\forecasting\hh_projs')
ENGLAND_CODE = "E92000001"


# %%
def main():
    process_gor_projections()
    process_country_projections()


# %%
def process_gor_projections() -> None:
    # Note only England has projections by sub areas
    # however both Scotland and Wales are themselves regions
    regional_forecast_filepath = (
        POPULATION_DIR / "2018_based_england_regions_pop_projections.xlsx"
    )

    gor_df = pd.read_excel(regional_forecast_filepath, sheet_name="Males", skiprows=6)
    male_df = convert_rgn_forecasts_to_segmentations(df=gor_df, g_seg=1)
    male_df_rgn = male_df[male_df["RGN2021"] != "E92000001"]
    male_df_country = create_english_totals_from_regional(df=male_df)

    gor_df = pd.read_excel(regional_forecast_filepath, sheet_name="Females", skiprows=6)
    female_df = convert_rgn_forecasts_to_segmentations(df=gor_df, g_seg=2)
    female_df_rgn = female_df[female_df["RGN2021"] != "E92000001"]
    female_df_country = create_english_totals_from_regional(df=female_df)

    scotland_forecast = POPULATION_DIR / "2022_based_scotland_pop_projections.xlsx"
    df_age_g_scotland = process_to_segmentations(path_in=scotland_forecast)
    df_age_g_scotland["RGN2021"] = "S92000003"

    wales_forecast = POPULATION_DIR / "2021_based_interim_wales_pop_projections.xlsx"
    df_age_g_wales = process_to_segmentations(path_in=wales_forecast)
    df_age_g_wales["RGN2021"] = "W92000004"

    gor_df = pd.concat([male_df_rgn, female_df_rgn, df_age_g_scotland, df_age_g_wales])
    country_df = pd.concat(
        [male_df_country, female_df_country, df_age_g_scotland, df_age_g_wales]
    )

    output_years = [year for year in FORECAST_YEARS if year in gor_df]

    for year in output_years:

        df_gor_wide = pd.pivot(
            gor_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
        )

        filestem = f"2018_20_21_regions_pop_projections_{year}"

        pp.save_preprocessed_hdf(
            source_file_path=POPULATION_DIR / f"{filestem}.hdf", df=df_gor_wide
        )

        df_country_wide = pd.pivot(
            country_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
        )

        filestem = f"2018_20_21_country_pop_projections_{year}"

        pp.save_preprocessed_hdf(
            source_file_path=POPULATION_DIR / f"{filestem}.hdf", df=df_country_wide
        )


def create_english_totals_from_regional(df: pd.DataFrame) -> pd.DataFrame:

    # find regions mentioned
    eng_regions_list = df["RGN2021"].unique().tolist()

    # remove england from the list
    eng_regions_list.remove(ENGLAND_CODE)
    eng_values = df[df["RGN2021"] == ENGLAND_CODE]

    regional_dfs = []
    for eng_region in eng_regions_list:
        df_current = eng_values.copy()
        df_current["RGN2021"] = eng_region
        regional_dfs.append(df_current)

    return pd.concat(regional_dfs)


def convert_rgn_forecasts_to_segmentations(
    df: pd.DataFrame, g_seg: int
) -> pd.DataFrame:

    df = df.rename(columns={"CODE": "RGN2021"})

    # have to limit forecast years to those available
    available_years = [year for year in FORECAST_YEARS if year in df.columns]
    keep_cols: list[int | str] = ["RGN2021", "AGE GROUP"]
    keep_cols.extend(available_years)
    df = df[keep_cols]

    # get lowest age
    df = allocate_age_ntem(df, available_years)

    df = df.groupby(["RGN2021", "age_ntem"]).sum()
    df["g"] = g_seg

    df = df.reset_index()
    return df


def allocate_age_ntem(df: pd.DataFrame, available_years: list[int]) -> pd.DataFrame:
    df["from_age"] = df["AGE GROUP"].str.split("-").str[0]

    # fix 15-19 issue mapping to two categories in ratio 1:4
    df_exc_15 = df[df["from_age"] != "15"].copy()
    # don't forget about 90+
    df_exc_15["from_age"] = df_exc_15["from_age"].str.replace("+", "")

    # drop All Ages
    df_exc_15 = df_exc_15[df_exc_15["from_age"] != "All ages"]
    # convert to int
    df_exc_15["from_age"] = df_exc_15["from_age"].astype(int)

    df_15 = df[df["from_age"] == "15"].copy()
    df_15["from_age"] = 15
    df_15[available_years] = df_15[available_years] * 0.2

    df_16 = df[df["from_age"] == "15"].copy()
    df_16["from_age"] = 16
    df_16[available_years] = df_16[available_years] * 0.8

    # and combine
    df = pd.concat([df_exc_15, df_15, df_16])

    df = find_age_ntem_enum(df=df, age_col="from_age")

    df = df.drop(columns=["AGE GROUP", "from_age"])

    return df


# %%
def find_age_ntem_enum(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df.loc[df[age_col] < 16, "age_ntem"] = 1  # aged 15 years and under
    df.loc[(df[age_col] >= 16) & (df[age_col] <= 74), "age_ntem"] = (
        2  # aged 16 to 74 years
    )
    df.loc[df[age_col] >= 75, "age_ntem"] = 3  # aged 75 and over
    df["age_ntem"] = df["age_ntem"].astype(int)
    return df


def fetch_region_correspondence() -> pd.DataFrame:

    return pd.read_csv(
        Path(
            r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
            "GOR2021_CD_NM_EWS.csv",
        )
    )


# %%
def process_country_projections() -> None:

    country_forecasts = {}

    england_forecast = (
        POPULATION_DIR / "2021_based_interim_england_pop_projections.xlsx"
    )
    df_age_g_eng = process_to_segmentations(path_in=england_forecast)
    country_forecasts["england"] = df_age_g_eng

    # note this is 2022 whereas england and wales are 2021
    scotland_forecast = POPULATION_DIR / "2022_based_scotland_pop_projections.xlsx"
    df_age_g_scotland = process_to_segmentations(path_in=scotland_forecast)
    country_forecasts["scotland"] = df_age_g_scotland

    wales_forecast = POPULATION_DIR / "2021_based_interim_wales_pop_projections.xlsx"
    df_age_g_wales = process_to_segmentations(path_in=wales_forecast)
    country_forecasts["wales"] = df_age_g_wales

    region_correspondence = pd.read_csv(
        Path(
            r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
            "GOR2021_CD_NM_EWS.csv",
        )
    )
    region_codes = region_correspondence["RGN21CD"]

    regions_df = create_2022_regions_df(
        country_forecasts=country_forecasts,
        region_codes=region_codes,
    )

    save_2022_to_hdfs(regions_df=regions_df)


# %%
def process_to_segmentations(path_in: Path) -> pd.DataFrame:
    df = pd.read_excel(path_in, sheet_name="Population")

    # for our purposes we can map '105 - 109' and '110 and over' to 105

    df.loc[df["Age"] == "105 - 109", "Age"] = 105
    df.loc[df["Age"] == "110 and over", "Age"] = 105

    df["Age"] = df["Age"].astype(int)

    keep_cols: list[int | str] = ["Sex", "Age"]

    keep_cols.extend(FORECAST_YEARS)

    df = df[keep_cols]
    df = df.copy()

    df = find_age_ntem_enum(df=df, age_col="Age")

    # and now allocate g segment
    df.loc[df["Sex"] == "Males", "g"] = 1
    df.loc[df["Sex"] == "Females", "g"] = 2
    df["g"] = df["g"].astype(int)

    df = df.drop(columns=["Sex", "Age"])

    df = df.groupby(["age_ntem", "g"]).sum()
    return df.reset_index()


# %% region approach
def create_2022_regions_df(
    country_forecasts: dict[str, pd.DataFrame],
    region_codes: list[str],
) -> pd.DataFrame:

    # if region approach with multiple columns for the data then go down this path

    # note that we duplicate the values for england to each region, not ideal but
    # as only going to look for growth from 2023 should be fine as long
    # as we don't use this value directly, and do it consistently for the base and future year

    region_list = []
    for region_code in region_codes:
        if region_code.startswith("E"):
            current_df = country_forecasts["england"]
        elif region_code.startswith("S"):
            current_df = country_forecasts["scotland"]
        elif region_code.startswith("W"):
            current_df = country_forecasts["wales"]
        else:
            raise ValueError(f"Unable to process {region_code}")

        region_df = current_df.copy()
        region_df["RGN2021"] = region_code
        region_list.append(region_df)

    regions_df = pd.concat(region_list)

    return regions_df.reset_index()


# %%
def save_2022_to_hdfs(regions_df: pd.DataFrame) -> None:
    for year in FORECAST_YEARS:
        df_wide = pd.pivot(
            regions_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
        )

        filestem = f"2021_22_based_ews_pop_projections_{year}"
        df_wide.to_csv(POPULATION_DIR / "temp_outputs" / f"{filestem}.csv")

        pp.save_preprocessed_hdf(
            source_file_path=POPULATION_DIR / f"{filestem}.hdf", df=df_wide
        )


# %%
if __name__ == "__main__":
    main()


def process_and_save_hh_projections() -> None:
    """
    Function to read in and pre-process the 2018-based ONS households
    projections (up to 2043)
    Outputs the totals household projections by GOR for each year, as hdfs

    Can substitute other sources of data for this input (e.g. NTEM) if required
    """
    rgn_corr = fetch_region_correspondence()

    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / 'england_LAs_2018_based_hh_projections.xlsx',
        sheet_name='406', header=4
    ).rename(columns={'Area code': 'region'})
    # Filter to the defined region and relevant years (2018-based ONS data is available up to 2043)
    hh_eng = hh_eng[hh_eng['Area name'].isin(rgn_corr['RGN21NM'].tolist())]
    hh_eng = hh_eng[['region'] + FORECAST_YEARS[0:5]]

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / 'scotland_2018-based_hh_projections.xlsx',
        sheet_name='Table 1', header=3, nrows=1
    )
    hh_scot = hh_scot.iloc[:, 1: 27]
    hh_scot.columns = hh_scot.columns.astype(int)
    # Define region code and filter to relevant years
    hh_scot['region'] = 'S92000003'
    hh_scot = hh_scot[['region'] + FORECAST_YEARS[0:5]]

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / 'wales_2018_based_hh_projections.csv', header=5, nrows=1
    )
    hh_wales = hh_wales.drop(hh_wales.columns[0: 2], axis=1)
    hh_wales.columns = hh_wales.columns.str.strip().astype(int)
    # Define region code and filter to relevant years
    hh_wales['region'] = 'W92000004'
    hh_wales = hh_wales[['region'] + FORECAST_YEARS[0:5]]

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    for year in FORECAST_YEARS[0:5]:
        hh_by_year = hh_projs.copy()
        hh_by_year = hh_by_year[['region', year]]

        # Create column to use as segmentation totals
        hh_by_year['total'] = 1

        df_wide = pd.pivot(
            hh_by_year, index=['total'], columns=['region'], values=year
        )

        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / 'hh_totals.hdf', df=df_wide, multiple_output_ref=str(year)
        )


def process_and_save_hh_projections_children() -> None:
    """
    Function to read in and pre-process the 2018-based ONS households
    projections (up to 2043)
    Outputs the children household projections by England GOR for each year, as hdfs
    1: Household with no children or all children non-dependent
    2: Household with one or more dependent children

    Can substitute other sources of data for this input (e.g. NTEM) if required

    """
    rgn_corr = fetch_region_correspondence()
    years_str = list(map(str, FORECAST_YEARS))

    # Household projections 2023 to 2043 (all years)
    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / 'eng_2018_based_Stage 2 projected households - Principal.xlsx',
        sheet_name='Households by type')

    # Filter to the regions, "all ages" and relevant years
    hh_eng = hh_eng[hh_eng['AREA NAME'].isin(rgn_corr['RGN21NM'].tolist())]
    hh_eng = hh_eng.loc[hh_eng['AGE GROUP'] == 'All ages']
    hh_eng = hh_eng.loc[hh_eng['HOUSEHOLD TYPE'] != 'Total']
    hh_eng = hh_eng[['CODE', 'HOUSEHOLD TYPE'] + FORECAST_YEARS[0:5]]

    # Map to Land Use children segmentation (hh with no children / hh with 1+ children)
    children_map_eng = {'One person households: Male': 1,
                        'One person households: Female': 1,
                        'Households with one dependent child': 2,
                        'Households with two dependent children': 2,
                        'Households with three or more dependent children': 2,
                        'Other households with two or more adults': 1}
    hh_eng['segment'] = hh_eng['HOUSEHOLD TYPE'].map(children_map_eng).astype(int)
    hh_eng = hh_eng.groupby(by=['CODE', 'segment'],
                            as_index=False)[FORECAST_YEARS[0:5]].sum()
    hh_eng.columns = hh_eng.columns.astype(str)

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / 'scotland_2018-based_hh_projections.xlsx',
        sheet_name='Table 2', header=3, nrows=7)
    hh_scot = hh_scot.iloc[:, 1: 28]

    # Define region code and filter to relevant years
    hh_scot['CODE'] = 'S92000003'
    hh_scot = hh_scot[['CODE', 'Household type'] + years_str[0:5]]

    # Map to Land Use children segmentation (hh with no children / hh with 1+ children)
    children_map_scot = {'1 adult female': 1, '1 adult male': 1, '2 adults': 1,
                         '1 adult, 1 child': 2, '1 adult, 2+ children': 2,
                         '2+ adult 1+ children': 2, '3+ person all adult': 1}
    hh_scot['segment'] = hh_scot['Household type'].map(children_map_scot).astype(int)
    hh_scot = hh_scot.groupby(by=['CODE', 'segment'],
                              as_index=False)[years_str[0:5]].sum()

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / 'wales_2018_based_hh_projections.csv', header=5, nrows=13
    )
    hh_wales = (hh_wales.drop(hh_wales.columns[0], axis=1).drop(hh_wales.index[0], axis=0).
                rename(columns={hh_wales.columns[1]: 'Household type'}))
    hh_wales.columns = hh_wales.columns.str.strip()
    hh_wales['Household type'] = hh_wales['Household type'].str.rstrip()

    # Define region code
    hh_wales['CODE'] = 'W92000004'
    # Filter to relevant years
    columns_to_keep = ['CODE', 'Household type'] + years_str[0:5]
    hh_wales = hh_wales[columns_to_keep]

    # Map to Land Use children segmentation
    # (hh with no children / hh with 1+ children)
    children_map_wales = {'1 person': 1, '2 person (No children)': 1, '2 person (1 adult, 1 child)': 2,
                          '3 person (No children)': 1, '3 person (2 adults, 1 child)': 2,
                          '3 person (1 adult, 2 children)': 2, '4 person (No children)': 1,
                          '4 person (2+ adults, 1+ children)': 2, '4 person (1 adult, 3 children)': 2,
                          '5+ person (No children)': 1, '5+ person (2+ adults, 1+ children)': 2,
                          '5+ person (1 adult, 4+ children)': 2}
    hh_wales['segment'] = hh_wales['Household type'].map(children_map_wales).astype(int)
    hh_wales = hh_wales.groupby(by=['CODE', 'segment'],
                                as_index=False)[years_str[0:5]].sum()
    hh_wales.columns = hh_eng.columns.astype(str)

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    for year in FORECAST_YEARS[0:5]:
        year = str(year)
        hh_year = hh_projs.copy()
        hh_year = hh_year[['CODE', 'segment', year]]

        # Into a wide format for DVector
        hh_year = pp.pivot_to_dvector(
            data=hh_year, zoning_column='CODE', index_cols=['segment'], value_column=year
        )
        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / 'hh_children.hdf', df=hh_year, multiple_output_ref=year
        )
