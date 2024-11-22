from pathlib import Path

import geopandas as gpd
import pandas as pd

RUN_DIR = Path(
    r"F:\Working\Land-Use\OUTPUTS_base_employment_bres_approach_a_weighting_2_level_check"
)
ASSURANCE_DIR = RUN_DIR / "03_Assurance"
SHP_DIR = Path(r"F:\Working\Land-Use\SHAPEFILES")
GEO_LU = Path(
    r"F:\Working\Land-Use\SHAPEFILES\CORRESPONDENCES\EW_LSOA21_TO_MSOA21_TO_LAD21_TO_GOR21.csv"
)
UNADJ_NAME = "e5"
ADJ_NAME = "e6"

FROM_TO = f"{UNADJ_NAME}_{ADJ_NAME}"

# these are definied by the relavant yml file for the restribution vector
SELECTED_REGIONS = ["North East", "North West", "Yorkshire and The Humber"]
SICS_THAT_CHANGE = [-1, 3, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18]


def main():
    compare_jobs_for_sic_soc()
    df = pd.read_csv(ASSURANCE_DIR / f"lsoa_{FROM_TO}_sic_soc_compare.csv")
    create_sic_soc_shp(df=df, geography="lsoa")
    lsoa_sic_soc_df = pd.read_csv(ASSURANCE_DIR / "sic_soc_compare.csv")
    msoa_sic_soc_df = process_lsoa_to_msoa(lsoa_df=lsoa_sic_soc_df)
    msoa_sic_soc_df.to_csv(
        ASSURANCE_DIR / f"msoa_{FROM_TO}_sic_soc_compare.csv", index=False
    )
    df = pd.read_csv(ASSURANCE_DIR / f"msoa_{FROM_TO}_sic_soc_compare.csv")
    create_sic_soc_shp(df=df, geography="msoa")

    create_xsoa_prop_of_lad()
    create_xsoa_lad_proportion_shp(geography="lsoa")
    create_xsoa_lad_proportion_shp(geography="msoa")


def compare_jobs_for_sic_soc() -> None:
    unadj_soc = pd.read_csv(ASSURANCE_DIR / f"{UNADJ_NAME}_soc_transposed.csv")
    adj_soc = pd.read_csv(ASSURANCE_DIR / f"{ADJ_NAME}_soc_transposed.csv")

    unadj_soc["all"] = unadj_soc["1"] + unadj_soc["2"] + unadj_soc["3"] + unadj_soc["4"]
    adj_soc["all"] = adj_soc["1"] + adj_soc["2"] + adj_soc["3"] + adj_soc["4"]

    unadj_soc = unadj_soc.rename(columns={"dz2011+lsoa2021_id": "lsoa"})
    adj_soc = adj_soc.rename(columns={"dz2011+lsoa2021_id": "lsoa"})

    unadj_soc = unadj_soc.rename(columns={"Unnamed: 0": "lsoa"})
    adj_soc = adj_soc.rename(columns={"Unnamed: 0": "lsoa"})

    unadj_sic = pd.read_csv(ASSURANCE_DIR / f"{UNADJ_NAME}_sic_transposed.csv")
    adj_sic = pd.read_csv(ASSURANCE_DIR / f"{ADJ_NAME}_sic_transposed.csv")

    unadj_sic = unadj_sic.rename(columns={"dz2011+lsoa2021_id": "lsoa"})
    adj_sic = adj_sic.rename(columns={"dz2011+lsoa2021_id": "lsoa"})

    unadj_sic = unadj_sic.rename(columns={"Unnamed: 0": "lsoa"})
    adj_sic = adj_sic.rename(columns={"Unnamed: 0": "lsoa"})

    ew_lsoas = pd.read_csv(GEO_LU)

    filtered_to_regions = ew_lsoas[ew_lsoas["RGN21NM"].isin(SELECTED_REGIONS)]

    lsoas_in_selected_regions = filtered_to_regions["LSOA21CD"].tolist()

    soc_compare = pd.merge(
        unadj_soc,
        adj_soc,
        how="outer",
        on="lsoa",
        suffixes=[f"_{UNADJ_NAME}", f"_{ADJ_NAME}"],
    )

    soc_compare = soc_compare[soc_compare["lsoa"].isin(lsoas_in_selected_regions)]

    soc_opts = ["1", "2", "3", "4"]

    soc_opts.append("all")

    for soc_opt in soc_opts:
        soc_compare[f"{soc_opt}_diff"] = (
            soc_compare[f"{soc_opt}_{UNADJ_NAME}"]
            - soc_compare[f"{soc_opt}_{ADJ_NAME}"]
        )

    sic_compare = pd.merge(
        unadj_sic,
        adj_sic,
        how="outer",
        on="lsoa",
        suffixes=[f"_{UNADJ_NAME}", f"_{ADJ_NAME}"],
    )
    sic_compare = sic_compare[sic_compare["lsoa"].isin(lsoas_in_selected_regions)]

    for sic_opt in SICS_THAT_CHANGE:
        sic_compare[f"{sic_opt}_diff"] = (
            sic_compare[f"{sic_opt}_{ADJ_NAME}"]
            - sic_compare[f"{sic_opt}_{UNADJ_NAME}"]
        )

    sic_compare.columns = "sic_" + sic_compare.columns
    soc_compare.columns = "soc_" + soc_compare.columns

    sic_compare = sic_compare.rename(columns={"sic_lsoa": "lsoa"})
    soc_compare = soc_compare.rename(columns={"soc_lsoa": "lsoa"})

    sic_soc_compare = pd.merge(sic_compare, soc_compare, how="outer")

    sic_soc_compare.to_csv(ASSURANCE_DIR / "sic_soc_compare.csv", index=False)

    # TODO: why is this written out again? Are the two csvs essentially the same?

    for col in sic_soc_compare.columns:
        try:
            sic_soc_compare[col] = sic_soc_compare[col].astype(int)
        except:
            ValueError

    out_filename = f"lsoa_{FROM_TO}_sic_soc_compare.csv"
    sic_soc_compare.to_csv(ASSURANCE_DIR / out_filename, index=False)


def create_sic_soc_shp(df: pd.DataFrame, geography: str) -> None:

    df = df.rename(columns={geography.lower(): f"{geography.upper()}21CD"})

    shp = get_northern_shp_for_geography(geography=geography)

    shp_with_values = shp.merge(df, on=f"{geography}21CD")

    for col in shp_with_values.columns:
        new_string = col.replace("_", "")
        shp_with_values = shp_with_values.rename(columns={col: new_string})

    shp_with_values.to_file(
        ASSURANCE_DIR / f"{geography.lower()}_with_{FROM_TO}.shp",
        driver="ESRI Shapefile",
    )


def process_lsoa_to_msoa(lsoa_df: pd.DataFrame) -> pd.DataFrame:

    lu = pd.read_csv(GEO_LU, usecols=["LSOA21CD", "MSOA21CD"])

    lu = lu.rename(columns={"LSOA21CD": "lsoa"})

    sic_soc_compare_with_msoa = pd.merge(lsoa_df, lu, on="lsoa")

    sic_soc_compare_with_msoa = sic_soc_compare_with_msoa.drop(columns=["lsoa"])

    if "lad21cd" in sic_soc_compare_with_msoa:
        sic_soc_compare_with_msoa = sic_soc_compare_with_msoa.drop(columns=["lad21cd"])

    msoa_values = sic_soc_compare_with_msoa.groupby(["MSOA21CD"]).sum().reset_index()

    for col in msoa_values.columns:
        try:
            msoa_values[col] = msoa_values[col].astype(int)
        except:
            ValueError

    return msoa_values


def create_xsoa_prop_of_lad():
    lsoa_by_category_file = f"lsoa_{FROM_TO}_sic_soc_compare.csv"
    df = pd.read_csv(ASSURANCE_DIR / lsoa_by_category_file)
    df = df.rename(
        columns={
            f"soc_all_{UNADJ_NAME}": f"all_{UNADJ_NAME}",
            f"soc_all_{ADJ_NAME}": f"all_{ADJ_NAME}",
        }
    )
    # attach lad to file
    lsoa_to_lad_lu = pd.read_csv(GEO_LU, usecols=["LSOA21CD", "LAD21CD"])
    lsoa_to_lad_lu = lsoa_to_lad_lu.rename(
        columns={"LSOA21CD": "lsoa", "LAD21CD": "lad21cd"}
    )
    lsoa_df = pd.merge(df, lsoa_to_lad_lu)

    lsoa_df_no_prop = lsoa_df.copy()

    make_proportions_df(df=lsoa_df, geography="lsoa")

    # make msoa df
    msoa_df = process_lsoa_to_msoa(lsoa_df=lsoa_df_no_prop)

    msoa_df = msoa_df.rename(columns={"MSOA21CD": "msoa"})

    msoa_to_lad_lu = pd.read_csv(GEO_LU, usecols=["MSOA21CD", "LAD21CD"])
    msoa_to_lad_lu = msoa_to_lad_lu.rename(
        columns={"MSOA21CD": "msoa", "LAD21CD": "lad21cd"}
    )
    msoa_to_lad_lu = msoa_to_lad_lu.drop_duplicates()

    msoa_df = pd.merge(msoa_df, msoa_to_lad_lu)

    make_proportions_df(df=msoa_df, geography="msoa")


def make_proportions_df(df: pd.DataFrame, geography: str) -> None:

    col_prefixes = ["all", "sic_7", "sic_16"]

    for col_prefix in col_prefixes:
        unadj_prop_col = f"{col_prefix}_{UNADJ_NAME}_%"
        adj_prop_col = f"{col_prefix}_{ADJ_NAME}_%"

        df[unadj_prop_col] = df.groupby(["lad21cd"])[
            f"{col_prefix}_{UNADJ_NAME}"
        ].transform(lambda x: x / x.sum())
        df[adj_prop_col] = df.groupby(["lad21cd"])[
            f"{col_prefix}_{ADJ_NAME}"
        ].transform(lambda x: x / x.sum())

        df[f"{col_prefix}_diff_%"] = df[adj_prop_col] - df[unadj_prop_col]
        df = df.round({unadj_prop_col: 5, adj_prop_col: 5, f"{col_prefix}_diff_%": 5})

    keep_cols = [geography, "lad21cd"]

    for col in df.columns:
        if col.endswith("%"):
            keep_cols.append(col)

    df = df[keep_cols]

    df.to_csv(ASSURANCE_DIR / f"{geography}_{FROM_TO}_lad_proportions.csv", index=False)


def create_xsoa_lad_proportion_shp(geography: str) -> None:

    geography_upper = geography.upper()
    geography_lower = geography.lower()

    shp = get_northern_shp_for_geography(geography=geography)

    df_props = pd.read_csv(
        ASSURANCE_DIR / f"{geography_lower}_{FROM_TO}_lad_proportions.csv"
    )
    df_props = df_props.drop(columns="lad21cd")

    df_props = df_props.rename(columns={geography_lower: f"{geography_upper}21CD"})

    shp_with_values = shp.merge(df_props, on=f"{geography_upper}21CD")

    for col in shp_with_values.columns:
        new_col = col.replace("_", "")
        shp_with_values = shp_with_values.rename(columns={col: new_col})

    shp_with_values.to_file(
        ASSURANCE_DIR / f"{geography_lower}_{FROM_TO}_lad_proportions.shp",
        driver="ESRI Shapefile",
    )


def get_northern_shp_for_geography(geography: str) -> gpd.GeoDataFrame:
    return gpd.read_file(
        SHP_DIR
        / f"{geography.upper()} (2021)"
        / f"{geography.upper()}_2021_EnglandWales_NORTH.shp"
    )


if __name__ == "__main__":
    main()
