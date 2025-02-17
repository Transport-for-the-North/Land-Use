# %%
from pathlib import Path

import pandas as pd

import land_use.preprocessing as pp


FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048]
ENGLISH_REGIONS_TO_CODES = ["", ""]
BASE_YEAR = 2023

population_dir = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs")


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

    df.loc[df["Age"] < 16, "ntem_age"] = 1  # aged 15 years and under
    df.loc[(df["Age"] >= 16) & (df["Age"] <= 74), "ntem_age"] = 2  # aged 16 to 74 years
    df.loc[df["Age"] >= 75, "ntem_age"] = 3  # aged 75 and over
    df["ntem_age"] = df["ntem_age"].astype(int)

    # and now gender
    df.loc[df["Sex"] == "Males", "g"] = 1
    df.loc[df["Sex"] == "Females", "g"] = 2
    df["g"] = df["g"].astype(int)

    df = df.drop(columns=["Sex", "Age"])

    return df.groupby(["ntem_age", "g"]).sum()


england_forecast = population_dir / "2021_based_interim_england_pop_projections.xlsx"
df_age_g_eng = process_to_segmentations(path_in=england_forecast)

# note this is 2022 whereas england and wales are 2021
scotland_forecast = population_dir / "2022_based_scotland_pop_projections.xlsx"
df_age_g_scotland = process_to_segmentations(path_in=scotland_forecast)

wales_forecast = population_dir / "2021_based_interim_wales_pop_projections.xlsx"
df_age_g_wales = process_to_segmentations(path_in=wales_forecast)


# %% region approach

# if region approach with mulitple columns for the data then go down this path

# note that we duplicate the values for england to each region, not ideal but
# as only going to look for growth from 2023 should be fine as long
# as we don't use this value directly, and do it consistently for the base and future year

region_correspondence = pd.read_csv(
    Path(
        r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
        "GOR2021_CD_NM_EWS.csv",
    )
)

# base_extract_eng_rgn = df_age_g[[BASE_YEAR]]
age_g_reset_eng = df_age_g_eng.copy().reset_index()
age_g_reset_scotland = df_age_g_scotland.copy().reset_index()
age_g_reset_wales = df_age_g_wales.copy().reset_index()


region_list = []
for region_code in region_correspondence["RGN21CD"]:
    if region_code.startswith("E"):
        current_df = age_g_reset_eng
    elif region_code.startswith("S"):
        current_df = age_g_reset_scotland
    elif region_code.startswith("W"):
        current_df = age_g_reset_wales
    else:
        raise ValueError(f"Unable to process {region_code}")

    region_df = current_df.copy()
    region_df["rgn_cd"] = region_code
    region_list.append(region_df)


regions_df = pd.concat(region_list)

regions_df = regions_df.reset_index()


for year in FORECAST_YEARS:
    df_wide = pd.pivot(
        regions_df, index=["ntem_age", "g"], columns=["rgn_cd"], values=year
    )

    filestem = f"2021_22_based_ews_pop_projections_{year}"
    df_wide.to_csv(population_dir / "temp_outputs" / f"{filestem}.csv")

    pp.save_preprocessed_hdf(
        source_file_path=population_dir / f"{filestem}.hdf", df=df_wide
    )


# The other key one is
# 2018_based_england_regions_pop_projections.xlsx
# which though older than above gives values by English GOR, which we can
# use to split down the English population 2021 forecast in future years
# here we will just get the values out.
