# %%
from pathlib import Path

import pandas as pd
import numpy as np

import land_use.preprocessing as pp

# Include the base year in here as we are pivoting from it
BASE_YEAR = 2023
CROSSOVER_YEAR = 2043
FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048, 2053]
POPULATION_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs")
HOUSEHOLDS_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\hh_projs")
OBR_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\OBR")
ENGLAND_CODE = "E92000001"

# %%
# mapping the various household compositions to the children segmentation
# 1: no children
# 2: 1 or more children

CHILDREN_MAPPING = {
    "One person households: Male": 1,
    "One person households: Female": 1,
    "Households with one dependent child": 2,
    "Households with two dependent children": 2,
    "Households with three or more dependent children": 2,
    "Other households with two or more adults": 1,
    "1 adult female": 1,
    "1 adult male": 1,
    "2 adults": 1,
    "1 adult, 1 child": 2,
    "1 adult, 2+ children": 2,
    "2+ adult 1+ children": 2,
    "3+ person all adult": 1,
    "1 person": 1,
    "2 person (No children)": 1,
    "2 person (1 adult, 1 child)": 2,
    "3 person (No children)": 1,
    "3 person (2 adults, 1 child)": 2,
    "3 person (1 adult, 2 children)": 2,
    "4 person (No children)": 1,
    "4 person (2+ adults, 1+ children)": 2,
    "4 person (1 adult, 3 children)": 2,
    "5+ person (No children)": 1,
    "5+ person (2+ adults, 1+ children)": 2,
    "5+ person (1 adult, 4+ children)": 2
}

# %%
def main():
    # for forecast_year in range(BASE_YEAR, 2054):
    #     write_ons_pop_growth_factors_from_base(
    #         base_year=BASE_YEAR, forecast_year=forecast_year
    #     )
    # process_and_save_hh_projections_children()
    #process_and_save_projections_1_adult_hhs()
    process_and_save_hh_projections()


def write_ons_pop_growth_factors_from_base(base_year: int, forecast_year: int) -> None:

    if base_year < CROSSOVER_YEAR:
        ons_pop_base = fetch_ons_pop_forecasts_up_to_crossover(year=base_year)
    else:
        ons_pop_base = fetch_ons_pop_forecasts_post_crossover(year=base_year)

    if forecast_year < CROSSOVER_YEAR:
        ons_pop_fy = fetch_ons_pop_forecasts_up_to_crossover(year=forecast_year)
    else:
        ons_pop_fy = fetch_ons_pop_forecasts_post_crossover(year=forecast_year)

    pop_growth_factor = ons_pop_fy / ons_pop_base

    filestem = f"pop_growth_factors_from_{base_year}"

    pp.save_preprocessed_hdf(
        source_file_path=POPULATION_DIR / f"{filestem}.hdf",
        df=pop_growth_factor,
        key=f"factors_from_{base_year}_to_{forecast_year}",
        mode="r+"
    )


# %%
def fetch_ons_pop_forecasts_up_to_crossover(year: int) -> pd.DataFrame:

    gor_2018, country_2018 = fetch_2018_pop_projections(year=year)
    country_2022 = fetch_2022_pop_projections(year=year)

    # adjust based on more recent country level data
    update_forecast_factor = country_2022 / country_2018
    return gor_2018 * update_forecast_factor


def fetch_ons_pop_forecasts_post_crossover(year: int) -> pd.DataFrame:
    # As regional 2018 forecasts aren't available after 2043 then a different approach is needed
    gor_2018_crossover, country_2018_crossover = fetch_2018_pop_projections(
        year=CROSSOVER_YEAR
    )
    country_2022_crossover = fetch_2022_pop_projections(year=CROSSOVER_YEAR)
    country_2022_fy = fetch_2022_pop_projections(year=year)

    # adjust from crossover to forecast year
    update_forecast_factor = country_2022_crossover / country_2018_crossover
    crossover_to_fy_factor = country_2022_fy / country_2022_crossover
    return gor_2018_crossover * update_forecast_factor * crossover_to_fy_factor


# %%
def fetch_2018_pop_projections(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Note only England has projections by sub areas
    # however both Scotland and Wales are themselves regions
    eng_regional_forecast_filepath = (
        POPULATION_DIR / "2018_based_england_regions_pop_projections.xlsx"
    )

    gor_df = pd.read_excel(
        eng_regional_forecast_filepath, sheet_name="Males", skiprows=6
    )
    male_df = convert_rgn_forecasts_to_segmentations(df=gor_df, g_seg=1)
    male_df_rgn = male_df[male_df["RGN2021"] != "E92000001"]
    male_df_country = create_english_totals_from_regional(df=male_df)

    gor_df = pd.read_excel(
        eng_regional_forecast_filepath, sheet_name="Females", skiprows=6
    )
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

    if not year in gor_df.columns:
        raise NotImplementedError(f"{year} is not included in dataset")

    df_gor_wide = pd.pivot(
        gor_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
    )

    df_country_wide = pd.pivot(
        country_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
    )

    return df_gor_wide, df_country_wide


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

    df = df.drop(columns="AREA")

    # get lowest age
    df = allocate_age_ntem(df)

    df = df.groupby(["RGN2021", "age_ntem"]).sum()
    df["g"] = g_seg

    df = df.reset_index()
    return df


def allocate_age_ntem(df: pd.DataFrame) -> pd.DataFrame:
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

    available_years = list(range(2018, 2044))
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
def fetch_2022_pop_projections(year: int) -> pd.DataFrame:

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

    country_pop_projections = pd.pivot(
        regions_df, index=["age_ntem", "g"], columns=["RGN2021"], values=year
    )

    return country_pop_projections


# %%
def process_to_segmentations(path_in: Path) -> pd.DataFrame:
    df = pd.read_excel(path_in, sheet_name="Population")

    # for our purposes we can map '105 - 109' and '110 and over' to 105

    df.loc[df["Age"] == "105 - 109", "Age"] = 105
    df.loc[df["Age"] == "110 and over", "Age"] = 105

    df["Age"] = df["Age"].astype(int)

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



def process_and_save_hh_projections() -> None:
    """
    Function to read in and pre-process the 2018-based ONS households
    projections
    Outputs the totals household projections by GOR for each year, as hdfs

    Can substitute other sources of data for this input (e.g. NTEM) if required
    """
    rgn_corr = fetch_region_correspondence()

    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / "england_LAs_2018_based_hh_projections.xlsx",
        sheet_name="406",
        header=4,
    ).rename(columns={"Area code": "region"})
    hh_eng = hh_eng.iloc[:, np.r_[0:2, 19:45]]
    # Filter to the defined regions (2018-based ONS data is available up to 2043)
    hh_eng = hh_eng[hh_eng["Area name"].isin(rgn_corr["RGN21NM"].tolist())].drop(
        columns=["Area name"]
    )

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / "scotland_2018-based_hh_projections.xlsx",
        sheet_name="Table 1",
        header=3,
        nrows=1,
    )
    hh_scot = hh_scot.iloc[:, 1:27]
    hh_scot.columns = hh_scot.columns.astype(int)
    # Define region code
    hh_scot["region"] = "S92000003"

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / "wales_2018_based_hh_projections.csv", header=5, nrows=1
    )
    hh_wales = hh_wales.drop(hh_wales.columns[0:2], axis=1)
    hh_wales.columns = hh_wales.columns.str.strip().astype(int)
    # Define region code
    hh_wales["region"] = "W92000004"

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    # Interpolate for any other years we want
    years = [int(yr) for yr in hh_projs.columns if str(yr).isnumeric()]
    max_year = max(years)
    min_year = min(years)
    for year in FORECAST_YEARS:
        print(year)
        if year in hh_projs.columns:
            # column already exists so nothing to do
            pass
        elif year > max_year:
            # As 2043 is the maximum year in the 2018-based ONS, continue the trend from 2038 to 2043
            year_a = 2038
            year_b = 2043
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            hh_projs[year] = perc_of_a * hh_projs[year_a] + perc_of_b * hh_projs[year_b]
        else:
            # before first year, raise error for now
            raise ValueError(
                f"Unable to extrapolate for {year}, earliest year is {min_year}"
            )

    # Export
    for year in FORECAST_YEARS:

        hh_projs[f"factors_from_{BASE_YEAR}"] = hh_projs[year] / hh_projs[BASE_YEAR]

        # Create column to use as segmentation totals
        hh_projs["total"] = 1

        df_wide = pd.pivot(hh_projs, index=["total"], columns=["region"], values=f"factors_from_{BASE_YEAR}")

        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / f"hh_totals_from_{BASE_YEAR}_factors.hdf",
            df=df_wide,
            key=f"factors_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )


def process_and_save_hh_projections_children() -> None:
    """
    Function to read in and pre-process the 2018-based ONS households
    projections
    Outputs the children household projections by England GOR for each year, as hdfs
    1: Household with no children or all children non-dependent
    2: Household with one or more dependent children

    Can substitute other sources of data for this input (e.g. NTEM) if required

    """
    rgn_corr = fetch_region_correspondence()

    # Household projections 2023 to 2043 (all years)
    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / "eng_2018_based_Stage 2 projected households - Principal.xlsx",
        sheet_name="Households by type",
    )

    # Filter to the regions and "all ages"
    hh_eng = hh_eng[hh_eng["AREA NAME"].isin(rgn_corr["RGN21NM"].tolist())]
    hh_eng = hh_eng.loc[hh_eng["AGE GROUP"] == "All ages"]
    hh_eng = hh_eng.loc[hh_eng["HOUSEHOLD TYPE"] != "Total"]
    years = [year for year in range(2018, 2044)]
    hh_eng = hh_eng.loc[:, ["CODE", "HOUSEHOLD TYPE"] + years]

    # Map to Land Use children segmentation (hh with no children / hh with 1+ children)
    hh_eng["segment"] = hh_eng["HOUSEHOLD TYPE"].map(CHILDREN_MAPPING).astype(int)
    hh_eng = hh_eng.groupby(by=["CODE", "segment"], as_index=False)[years].sum()

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / "scotland_2018-based_hh_projections.xlsx",
        sheet_name="Table 2",
        header=3,
        nrows=7,
    )
    hh_scot = hh_scot.iloc[:, 1:28]
    hh_scot.columns = [int(col) if col.isdigit() else col for col in hh_scot.columns]
    hh_scot = hh_scot.loc[:, ["Household type"] + years]

    # Define region code
    hh_scot["CODE"] = "S92000003"

    # Map to Land Use children segmentation (hh with no children / hh with 1+ children)
    hh_scot["segment"] = hh_scot["Household type"].map(CHILDREN_MAPPING).astype(int)
    hh_scot = hh_scot.groupby(by=["CODE", "segment"], as_index=False)[years].sum()

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / "wales_2018_based_hh_projections.csv", header=5, nrows=13
    )
    hh_wales = (
        hh_wales.drop(hh_wales.columns[0], axis=1)
        .drop(hh_wales.index[0], axis=0)
        .rename(columns={hh_wales.columns[1]: "Household type"})
    )
    hh_wales.columns = hh_wales.columns.str.strip()
    hh_wales.columns = [int(col) if col.isdigit() else col for col in hh_wales.columns]
    hh_wales["Household type"] = hh_wales["Household type"].str.rstrip()
    # Define region code
    hh_wales["CODE"] = "W92000004"

    # Map to Land Use children segmentation
    # (hh with no children / hh with 1+ children)
    hh_wales["segment"] = hh_wales["Household type"].map(CHILDREN_MAPPING).astype(int)
    hh_wales = hh_wales.groupby(by=["CODE", "segment"], as_index=False)[years].sum()

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    # Interpolate for any other years we want
    years = [int(yr) for yr in hh_projs.columns if str(yr).isnumeric()]
    max_year = max(years)
    min_year = min(years)
    for year in FORECAST_YEARS:
        print(year)
        if year in hh_projs.columns:
            # column already exists so nothing to do
            pass
        elif year > max_year:
            # As 2043 is the maximum year in the 2018-based ONS, continue the trend from 2038 to 2043
            year_a = 2038
            year_b = 2043
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            hh_projs[year] = perc_of_a * hh_projs[year_a] + perc_of_b * hh_projs[year_b]
        else:
            # before first year, raise error for now
            raise ValueError(
                f"Unable to extrapolate for {year}, earliest year is {min_year}"
            )

    hh_projs = hh_projs.rename(columns={"segment": "children"})
    hh_projs.columns = [str(col) for col in hh_projs.columns]
    
    for year in FORECAST_YEARS:

        hh_projs[f"factors_{year}"] = hh_projs[str(year)] / hh_projs[str(BASE_YEAR)]

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=hh_projs,
            zoning_column="CODE",
            index_cols=["children"],
            value_column=f"factors_{year}",
        )
        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / f"hh_children_from_{BASE_YEAR}_factors.hdf",
            df=df_wide,
            key=f"factors_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )


def pre_process_obr() -> None:
    # Read in OBR data (using quarter 1 data)
    df = pd.read_csv(OBR_INPUT_DIR / "obr_forecasts_feb_2025.csv")

    # employment
    emp = df.loc[
        (df["Measure"] == "Employment forecast")
        | (df["Measure"] == "Employment outturn")
    ]
    emp = emp[emp["Quarter"].str.endswith("Q1")].sort_values("Quarter")
    emp["year"] = emp["Quarter"].str.replace("Q1", "").astype(int)
    emp["Measure"] = "employment"

    # unemployment (percentage pf population 16+)
    unemp = df.loc[
        (df["Measure"] == "Unemployment rate forecast")
        | (df["Measure"] == "Unemployment rate outturn")
    ]
    unemp = unemp[unemp["Quarter"].str.endswith("Q1")].sort_values("Quarter")
    unemp["year"] = unemp["Quarter"].str.replace("Q1", "").astype(int)
    unemp["Measure"] = "unemployment"

    both = pd.concat([emp, unemp], ignore_index=False)[["year", "Measure", "Value"]]
    both = both.pivot(index="Measure", columns="year", values="Value")

    years = [int(yr) for yr in both.columns if str(yr).isnumeric()]
    min_year = min(years)
    # Calculate growth
    for year in FORECAST_YEARS:
        print(year)
        if year in both.columns:
            # column already exists
            # TODO check we just take the value?
            pass
        elif year < min_year:
            # before first year, raise error for now
            raise ValueError(
                f"Unable to extrapolate for {year}, earliest year is {min_year}"
            )
        else:
            # As 2030 is the maximum year, continue the 2023 to 2030
            year_a = 2023
            year_b = 2030
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            both[year] = perc_of_a * both[year_a] + perc_of_b * both[year_b]

    # Prepare for export
    export = both[FORECAST_YEARS]
    export.to_csv(
        OBR_INPUT_DIR / "preprocessing/OBR_forecasts_processed.csv",
        index=True,
        index_label="Measure",
    )


def process_and_save_hh_projections_adults_ss():
    # This function groups the data based on what is available and clearly defined
    # For example, the England data does not distinguish the number of adults in the children households, therefore
    # these use 1 adult households, and 2+ adult households, with children definitions ruled out as 0s

    # For Scotland and Wales, the children definitions can be used as these specify the number of adults, however
    # it is unclear on households between 2 adults and 3+ adults, so these use 1 adult and 2+ adult households

    rgn_corr = fetch_region_correspondence()

    # Household projections 2023 to 2043 (all years)
    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / "eng_2018_based_Stage 2 projected households - Principal.xlsx",
        sheet_name="Households by type",
    )

    # Filter to the regions and "all ages"
    hh_eng = hh_eng[hh_eng["AREA NAME"].isin(rgn_corr["RGN21NM"].tolist())]
    hh_eng = hh_eng.loc[hh_eng["AGE GROUP"] == "All ages"]
    hh_eng = hh_eng.loc[hh_eng["HOUSEHOLD TYPE"] != "Total"]
    years = [year for year in range(2018, 2044)]
    hh_eng = hh_eng.loc[:, ["CODE", "HOUSEHOLD TYPE"] + years]

    # Map to Land Use adults segmentation
    # (1 adult households / 2+ adult other households / 0 for children related segmentation)
    adults_map_eng = {
        "One person households: Male": 1,
        "One person households: Female": 1,
        "Households with one dependent child": 0,
        "Households with two dependent children": 0,
        "Households with three or more dependent children": 0,
        "Other households with two or more adults": 2
    }
    hh_eng["segment"] = hh_eng["HOUSEHOLD TYPE"].map(adults_map_eng).astype(int)
    hh_eng = hh_eng.groupby(by=["CODE", "segment"], as_index=False)[years].sum()
    # drop the children segments (0s)
    hh_eng = hh_eng.loc[hh_eng['segment'] != 0]
    # Use 2+ adults data for 3+ adults as this will just be used for calculating growth factors in main code
    eng_3_adults = hh_eng.copy()
    eng_3_adults = eng_3_adults.loc[hh_eng['segment'] == 2]
    eng_3_adults['segment'] = 3

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_DIR / "scotland_2018-based_hh_projections.xlsx",
        sheet_name="Table 2",
        header=3,
        nrows=7,
    )
    hh_scot = hh_scot.iloc[:, 1:28]
    hh_scot.columns = [int(col) if col.isdigit() else col for col in hh_scot.columns]
    hh_scot = hh_scot.loc[:, ["Household type"] + years]

    # Define region code
    hh_scot["CODE"] = "S92000003"

    # Map to Land Use adults segmentation
    # we have a direct correspondence between these except for "2+ adult 1+ children", as this could be 2 or 3+ adults
    # (1 adult households / 2+ adult other households)
    adults_map_scot = {
        "1 adult female": 1,
        "1 adult male": 1,
        "2 adults": 2,
        "1 adult, 1 child": 1,
        "1 adult, 2+ children": 1,
        "2+ adult 1+ children": 2,
        "3+ person all adult": 2,
    }
    hh_scot["segment"] = hh_scot["Household type"].map(adults_map_scot).astype(int)
    hh_scot = hh_scot.groupby(by=["CODE", "segment"], as_index=False)[years].sum()
    # Use 2+ adults data for 3+ adults as this will just be used for calculating growth factors in main code
    scot_3_adults = hh_scot.copy()
    scot_3_adults = scot_3_adults.loc[hh_scot['segment'] == 2]
    scot_3_adults['segment'] = 3

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_DIR / "wales_2018_based_hh_projections.csv", header=5, nrows=13
    )
    hh_wales = (
        hh_wales.drop(hh_wales.columns[0], axis=1)
        .drop(hh_wales.index[0], axis=0)
        .rename(columns={hh_wales.columns[1]: "Household type"})
    )
    hh_wales.columns = hh_wales.columns.str.strip()
    hh_wales.columns = [int(col) if col.isdigit() else col for col in hh_wales.columns]
    hh_wales["Household type"] = hh_wales["Household type"].str.rstrip()
    # Define region code
    hh_wales["CODE"] = "W92000004"

    # Map to Land Use children segmentation
    # we have a direct correspondence between these except for "2+ adult 1+ children", as this could be 2 or 3+ adults
    # (1 adult households / 2+ adult other households)
    adults_map_wales = {
        "1 person": 1,
        "2 person (No children)": 2,
        "2 person (1 adult, 1 child)": 1,
        "3 person (No children)": 2,
        "3 person (2 adults, 1 child)": 2,
        "3 person (1 adult, 2 children)": 1,
        "4 person (No children)": 2,
        "4 person (2+ adults, 1+ children)": 2,
        "4 person (1 adult, 3 children)": 1,
        "5+ person (No children)": 2,
        "5+ person (2+ adults, 1+ children)": 2,
        "5+ person (1 adult, 4+ children)": 1,
    }
    hh_wales["segment"] = hh_wales["Household type"].map(adults_map_wales).astype(int)
    hh_wales = hh_wales.groupby(by=["CODE", "segment"], as_index=False)[years].sum()
    # Use 2+ adults data for 3+ adults as this will just be used for calcualting growth factors in main code
    wales_3_adults = hh_wales.copy()
    wales_3_adults = wales_3_adults.loc[hh_wales['segment'] == 2]
    wales_3_adults['segment'] = 3

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, eng_3_adults, hh_scot, scot_3_adults, hh_wales, wales_3_adults])

    # Interpolate for any other years we want
    years = [int(yr) for yr in hh_projs.columns if str(yr).isnumeric()]
    max_year = max(years)
    min_year = min(years)
    for year in FORECAST_YEARS:
        print(year)
        if year in hh_projs.columns:
            # column already exists so nothing to do
            pass
        elif year > max_year:
            # As 2043 is the maximum year in the 2018-based ONS, continue the trend from 2038 to 2043
            year_a = 2038
            year_b = 2043
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            hh_projs[year] = perc_of_a * hh_projs[year_a] + perc_of_b * hh_projs[year_b]
        else:
            # before first year, raise error for now
            raise ValueError(f"Unable to extrapolate for {year}, earliest year is {min_year}")

        hh_year = hh_projs.copy()
        hh_year = hh_year[["CODE", "segment", year]].rename(columns={"segment": "adults"})
        hh_year.columns = [str(col) for col in hh_year.columns]

        # Into a wide format for DVector
        hh_year = pp.pivot_to_dvector(
            data=hh_year,
            zoning_column="CODE",
            index_cols=["adults"],
            value_column=str(year),
        )
        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / "hh_adults.hdf",
            df=hh_year,
            multiple_output_ref=str(year),
        )


def process_and_save_projections_1_adult_hhs():
    # This function processes the households data into 1 adult male and female households, by local authority (2019)

    rgn_corr = fetch_region_correspondence()
    years = [year for year in range(2018, 2044)]

    # -------------
    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_DIR / "eng_2018_based_Stage 2 projected households - Principal (2019 geog).xlsx",
        sheet_name="Households by type",
    )
    # Filter to the local authorities, "all ages" and 1-person households
    hh_eng = hh_eng[~hh_eng["AREA NAME"].isin(rgn_corr["RGN21NM"].tolist()) & ~hh_eng["AREA NAME"].isin(["England"])]
    hh_eng = hh_eng.loc[hh_eng["AGE GROUP"] == "All ages"]
    hh_eng = hh_eng.loc[(hh_eng["HOUSEHOLD TYPE"] == "One person households: Male") | (
            hh_eng["HOUSEHOLD TYPE"] == "One person households: Female")]
    hh_eng = hh_eng.loc[:, ["CODE", "HOUSEHOLD TYPE"] + years]

    # Map to Land Use gender segmentation
    hh_eng["g"] = hh_eng["HOUSEHOLD TYPE"].map({
        "One person households: Male": 1,
        "One person households: Female": 2
    }).astype(int)
    hh_eng = hh_eng.groupby(by=["CODE", "g"], as_index=False)[years].sum()

    # -------------
    # SCOTLAND
    scot_la_codes = pd.read_csv(
        HOUSEHOLDS_DIR / "scot_LA_codes.csv",
        usecols=["lad19cd", "lad19nm"]
    )
    scot_dfs = []
    for code in scot_la_codes["lad19nm"]:
        hh_scot = pd.read_excel(
            HOUSEHOLDS_DIR / "scotland_2018_based_hh_detailed-area-tables-principal-projection.xlsx",
            sheet_name=code,
            header=4,
            nrows=34,
        )
        hh_scot = hh_scot.rename(columns={hh_scot.columns[0]: "Household type", hh_scot.columns[1]: "age"})
        hh_scot = hh_scot.drop(hh_scot.columns[28: 34], axis=1)
        hh_scot = hh_scot[hh_scot["age"] != "All Ages"]
        hh_scot["region"] = code
        hh_scot["CODE"] = hh_scot["region"].map(scot_la_codes.set_index("lad19nm")["lad19cd"].to_dict())
        hh_scot["g"] = hh_scot["Household type"].map(
            {"1 person male": 1,
             "1 person female": 2}).astype(int)
        hh_scot.columns = [int(col) if col.isdigit() else col for col in hh_scot.columns]
        hh_scot = hh_scot.groupby(by=["CODE", "g"], as_index=False)[years].sum()
        scot_dfs.append(hh_scot)
    hh_scot = pd.concat(scot_dfs)

    # -------------
    # WALES
    wales_dfs = []
    for g in ["males", "females"]:
        hh_wales = pd.read_csv(
            HOUSEHOLDS_DIR / f"wales_2018_based_hh_projections_LAs_1_person_hh_{g}.csv", header=8, nrows=22,
        )
        hh_wales = (
            hh_wales.rename(columns={hh_wales.columns[0]: "Household type"})
        )
        hh_wales["Household type"] = hh_wales["Household type"].str.rstrip()

        wales_la_codes = pd.read_csv(
            HOUSEHOLDS_DIR / "wales_LA_codes.csv",
            usecols=["lad19cd", "lad19nm"]
        )
        hh_wales = pd.merge(
            hh_wales,
            wales_la_codes,
            left_on="Household type",
            right_on="lad19nm",
            how="left"
        ).rename(columns={"lad19cd": "CODE"})
        if g == "males":
            hh_wales["g"] = 1
        elif g == "females":
            hh_wales["g"] = 2
        wales_dfs.append(hh_wales)
    hh_wales = pd.concat(wales_dfs)
    hh_wales.columns = hh_wales.columns.str.strip()
    hh_wales.columns = [int(col) if col.isdigit() else col for col in hh_wales.columns]
    hh_wales = hh_wales.groupby(by=["CODE", "g"], as_index=False)[years].sum()

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    # Interpolate for any other years we want
    years = [int(yr) for yr in hh_projs.columns if str(yr).isnumeric()]
    max_year = max(years)
    min_year = min(years)
    for year in FORECAST_YEARS:
        print(year)
        # note this this forecast is provide for all years between minimum and maximum so don't need to interpolate between them
        if year in hh_projs.columns:
            # column already exists so nothing to do
            pass
        elif year > max_year:
            # As 2043 is the maximum year in the 2018-based ONS, continue the trend from 2038 to 2043
            year_a = 2038 # why are we doing 2038 here and not 2042?
            year_b = 2043
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            hh_projs[year] = perc_of_a * hh_projs[year_a] + perc_of_b * hh_projs[year_b]
        else:
            # before first year, raise error for now
            raise ValueError(f"Unable to extrapolate for {year}, earliest year is {min_year}")

    hh_year = hh_projs.copy()
    hh_year = hh_year[["CODE", "g", year]]
    hh_projs['adults'] = 1
    hh_projs.columns = [str(col) for col in hh_projs.columns]

    print(hh_year.columns)
    
    for year in FORECAST_YEARS:

        hh_projs[f"{year}_factor"] = hh_projs[str(year)] / hh_projs[str(BASE_YEAR)]

        # Into a wide format for DVector
        hh_projs_wide = pp.pivot_to_dvector(
            data=hh_projs,
            zoning_column="CODE",
            index_cols=["g", "adults"],
            value_column=f"{year}_factor",
        )
        pp.save_preprocessed_hdf(
            source_file_path=HOUSEHOLDS_DIR / f"hh_1_adult_by_g_from_{BASE_YEAR}_factors.hdf",
            df=hh_projs_wide,
            key=f"factors_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )


# %%
if __name__ == "__main__":
    main()
