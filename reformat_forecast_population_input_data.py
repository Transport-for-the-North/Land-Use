# %%
from pathlib import Path

import pandas as pd

import land_use.preprocessing as pp


FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048]
POPULATION_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs")


# %%
def main():
    process_2018_projections()
    process_2021_22_projections()


# %%
def process_2018_projections() -> None:
    # Note this is for england only
    forecast_filepath = (
        POPULATION_DIR / "2018_based_england_regions_pop_projections.xlsx"
    )
    male_df = convert_rgn_forecasts_to_segmentations(
        path_in=forecast_filepath, sheet_name="Males"
    )
    female_df = convert_rgn_forecasts_to_segmentations(
        path_in=forecast_filepath, sheet_name="Females"
    )

    df = pd.concat([male_df, female_df])
    df = df.reset_index()

    output_years = [year for year in FORECAST_YEARS if year in df]

    for year in output_years:

        df_wide = pd.pivot(df, index=["ntem_age", "g"], columns=["GOR"], values=year)

        filestem = f"2018_based_english_regions_pop_projections_{year}"

        pp.save_preprocessed_hdf(
            source_file_path=POPULATION_DIR / f"{filestem}.hdf", df=df_wide
        )


def convert_rgn_forecasts_to_segmentations(
    path_in: Path, sheet_name: str
) -> pd.DataFrame:

    df = pd.read_excel(path_in, sheet_name=sheet_name, skiprows=6)
    df = df.rename(columns={"CODE": "GOR"})

    region_correspondence = fetch_region_correspondence()
    region_codes = region_correspondence["RGN21CD"]
    df = df[df["GOR"].isin(region_codes)]

    # have to limit forecast years to those available
    available_years = [year for year in FORECAST_YEARS if year in df.columns]
    keep_cols: list[int | str] = ["GOR", "AGE GROUP"]
    keep_cols.extend(available_years)
    df = df[keep_cols]

    # get lowest age
    df = allocate_ntem_age(df, available_years)

    df = df.groupby(["GOR", "ntem_age"]).sum()

    if sheet_name == "Males":
        df["g"] = 1
    elif sheet_name == "Females":
        df["g"] = 2
    else:
        raise ValueError(f"unable to allocate g segment for {sheet_name}")

    return df


def allocate_ntem_age(df, available_years):
    df["from_age"] = df["AGE GROUP"].str.split("-").str[0]

    # fix 15-19 issue mapping to two categories in ratio 1:4
    df_exc_15 = df[df["from_age"] != "15"].copy()
    df_15 = df[df["from_age"] == "15"].copy()

    df_16 = df[df["from_age"] == "15"].copy()

    df_15[available_years] = df_15[available_years] / 0.2
    df_16[available_years] = df_16[available_years] / 0.8

    df = pd.concat([df_exc_15, df_15, df_16])

    # don't forget about 90+
    df["from_age"] = df["from_age"].str.replace("+", "")

    # drop All Ages
    df = df[df["from_age"] != "All ages"]

    df["from_age"] = df["from_age"].astype(int)

    df = find_age_ntem_enum(df=df, age_col="from_age")

    df = df.drop(columns=["AGE GROUP", "from_age"])

    return df


# %%
def find_age_ntem_enum(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    df.loc[df[age_col] < 16, "ntem_age"] = 1  # aged 15 years and under
    df.loc[(df[age_col] >= 16) & (df[age_col] <= 74), "ntem_age"] = (
        2  # aged 16 to 74 years
    )
    df.loc[df[age_col] >= 75, "ntem_age"] = 3  # aged 75 and over
    df["ntem_age"] = df["ntem_age"].astype(int)
    return df


def fetch_region_correspondence() -> pd.DataFrame:

    return pd.read_csv(
        Path(
            r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
            "GOR2021_CD_NM_EWS.csv",
        )
    )


# %%
def process_2021_22_projections() -> None:

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

    df = df.groupby(["ntem_age", "g"]).sum()
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
        region_df["rgn_cd"] = region_code
        region_list.append(region_df)

    regions_df = pd.concat(region_list)

    return regions_df.reset_index()


# %%
def save_2022_to_hdfs(regions_df: pd.DataFrame) -> None:
    for year in FORECAST_YEARS:
        df_wide = pd.pivot(
            regions_df, index=["ntem_age", "g"], columns=["rgn_cd"], values=year
        )

        filestem = f"2021_22_based_ews_pop_projections_{year}"
        df_wide.to_csv(POPULATION_DIR / "temp_outputs" / f"{filestem}.csv")

        pp.save_preprocessed_hdf(
            source_file_path=POPULATION_DIR / f"{filestem}.hdf", df=df_wide
        )


# %%
if __name__ == "__main__":
    main()
