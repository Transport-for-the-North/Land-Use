from pathlib import Path

import numpy as np
import pandas as pd

EMPLOYMENT_DISTRIBUTION_DIR = Path(
    r"I:\NorMITs Land Use\2023\import\Employment Attraction Distributions"
)
RAW_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "raw data"
INTERMIDIATE_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "intermediate steps"
INTERMIDIATE_DIR.mkdir(exist_ok=True)

FLOOR_TYPES = ["all", "industry", "office", "other", "retail"]


def main():
    infilled_lsoa_floorspaces = process_floorspaces()
    infilled_lsoa_rates = process_voa_rates()
    calc_values(floorspaces=infilled_lsoa_floorspaces, rates=infilled_lsoa_rates)


def process_floorspaces() -> pd.DataFrame:
    # Note the _1 which are floorspace
    filepaths = list(RAW_DIR.glob("table_*_1.csv"))

    # First get msoa sourted
    dfs_wide = extract_data_for_geography(filepaths=filepaths, geography="MSOA")
    for floor_type in FLOOR_TYPES:
        dfs_wide[floor_type] = np.where(dfs_wide["all"] == 0, 0, dfs_wide[floor_type])

    dfs_wide = infilling_msoa_all_for_2023(df=dfs_wide)
    infilled_msoa = infill_floorspace_msoa_floor_types_from_lad(df=dfs_wide)
    infilled_msoa.to_csv(INTERMIDIATE_DIR / "infilled_floorspace_msoa.csv", index=False)

    # And now LSOA which uses msoa level data so must follow
    dfs_wide = extract_data_for_geography(filepaths=filepaths, geography="LSOA")
    lsoa_w_msoa = attach_msoa_to_lsoa(df=dfs_wide)
    msoa_targets = calc_msoa_current_and_targets(
        lsoa_w_msoa=lsoa_w_msoa, infilled_msoa=infilled_msoa
    )
    lsoa_df = calc_lsoas_with_no_all_value(lsoa_with_msoa=lsoa_w_msoa)
    lsoa_w_msoa = infilling_lsoa_all_values(lsoa_df=lsoa_df, msoa_targets=msoa_targets)
    df = calc_msoa_floor_type_splits(
        lsoa_with_msoa=lsoa_w_msoa, infilled_msoa=infilled_msoa
    )
    df = infill_lsoa_floor_type_with_splits(df=df)

    lu = pd.read_csv(RAW_DIR / "ews_msoa_to_lad_lu.csv")
    lu.columns = ["msoa_code", "lad_code"]
    df = pd.merge(df, lu)

    df = df.rename(columns={"ons_code": "lsoa_code", "ons_name": "lsoa_name"})
    df.to_csv(INTERMIDIATE_DIR / "infilled_floorspaces_lsoa.csv", index=False)

    return df


def extract_data_for_geography(filepaths: list[Path], geography: str) -> pd.DataFrame:
    dfs_list = []
    for filepath in filepaths:
        if filepath.stem.endswith("_1"):
            measure = "floor_type"
        elif filepath.stem.endswith("_2"):
            measure = "voa_value"
        else:
            raise ValueError(f"unable to determine measure for {filepath.name}")
        df = extract_file_data(filepath=filepath, geography=geography, measure=measure)
        dfs_list.append(df)
    dfs = pd.concat(dfs_list)
    dfs_wide = dfs.pivot(index=["ons_code", "ons_name"], columns=measure, values="2023")
    dfs_wide = dfs_wide.astype(float)
    dfs_wide = dfs_wide.reset_index()

    return dfs_wide


def infilling_msoa_all_for_2023(df: pd.DataFrame) -> pd.DataFrame:
    # For 2023 MSOA there are only three msoas masked at the all floorspace level.
    # Having looked at the LAD totals for 'Swindon' and 'Chester West and Chester'
    # Then it is clear that the number must be small.
    # For simplicity have set them to 0
    # Swindon 018
    # Cheshire West and Chester 013
    # Swindon 026

    df["all"] = df["all"].fillna(0)
    return df


def infill_floorspace_msoa_floor_types_from_lad(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    lu = pd.read_csv(RAW_DIR / "ews_msoa_to_lad_lu.csv")
    lu.columns = ["ons_code", "lad_code"]

    df = pd.merge(df, lu, how="left", on="ons_code")

    for floor_type in FLOOR_TYPES:
        df[f"{floor_type}_average"] = df.groupby("lad_code")[floor_type].transform(
            lambda x: x.mean()
        )

    for floor_type in FLOOR_TYPES:
        df[f"{floor_type}_%"] = df[f"{floor_type}_average"] / df["all_average"]

        df[f"{floor_type}_infill"] = df[f"{floor_type}_%"] * df["all"]

        df[f"{floor_type}_working"] = df[floor_type]

        df[f"{floor_type}_working"] = df[f"{floor_type}_working"].fillna(
            df[f"{floor_type}_infill"]
        )

    # init
    df["factor"] = 1.0

    for loop_index in range(200):

        df = msoa_balance_loop(df=df)
        sig_diff = df[df["to_be_allocated"].abs() >= 3]

        sig_diff = sig_diff.copy()

        max_factor = max(sig_diff["factor"].max(), 1.0)
        min_factor = min(sig_diff["factor"].min(), 1.0)

    if len(sig_diff) > 0:
        print(
            f"After {loop_index+1} loops factors are, {max_factor:.2f} {min_factor:.2f}"
        )
    else:
        print("Balancing has converged")

    df.to_csv(INTERMIDIATE_DIR / "msoa_original_and_final_floorspace.csv", index=False)

    output_cols = [
        col for col in df.columns if col.startswith("ons") or col.endswith("_working")
    ]

    final_df = df[output_cols]

    # And remove _working from end of relevant columns

    col_renamer = {
        "all_working": "all",
        "industry_working": "industry",
        "office_working": "office",
        "other_working": "other",
        "retail_working": "retail",
    }
    final_df = final_df.rename(columns=col_renamer)

    return final_df


def calc_lsoas_with_no_all_value(lsoa_with_msoa: pd.DataFrame) -> pd.DataFrame:

    # way to distingush between where we have an na or not in
    lsoa_with_msoa["all_given"] = np.where(lsoa_with_msoa["all"].isna(), 0, 1)

    lsoa_with_msoa["lsoas_in_msoa"] = lsoa_with_msoa.groupby("msoa_code")[
        "all_given"
    ].transform("count")

    lsoa_with_msoa["lsoas_with_all_given"] = lsoa_with_msoa.groupby("msoa_code")[
        "all_given"
    ].transform("sum")

    lsoa_with_msoa["lsoas_missing_all_value"] = (
        lsoa_with_msoa["lsoas_in_msoa"] - lsoa_with_msoa["lsoas_with_all_given"]
    )

    return lsoa_with_msoa


def infilling_lsoa_all_values(
    lsoa_df: pd.DataFrame, msoa_targets: pd.DataFrame
) -> pd.DataFrame:

    # drop all to avoid clashing
    msoa_targets = msoa_targets.drop(columns="all")

    lsoa_with_msoa = pd.merge(lsoa_df, msoa_targets, how="left")

    lsoa_with_msoa["diff"] = lsoa_with_msoa["diff"].mask(lsoa_with_msoa["diff"] < 0, 0)

    lsoa_with_msoa["lsoa_all_infill_value"] = (
        lsoa_with_msoa["diff"] / lsoa_with_msoa["lsoas_missing_all_value"]
    )

    lsoa_with_msoa["all"] = lsoa_with_msoa["all"].fillna(
        lsoa_with_msoa["lsoa_all_infill_value"]
    )

    return lsoa_with_msoa


def calc_msoa_floor_type_splits(
    lsoa_with_msoa: pd.DataFrame, infilled_msoa: pd.DataFrame
) -> pd.DataFrame:

    # Use msoa level data to be able to work out splits where we haven't been given data at the lsoa level

    # sort out columns in msoa table to avoid clashes with the lsoa table
    infilled_msoa = infilled_msoa.add_prefix("msoa_")
    infilled_msoa = infilled_msoa.rename(
        columns={"msoa_ons_code": "msoa_code", "msoa_ons_name": "msoa_name"}
    )

    for floor_type in FLOOR_TYPES:
        infilled_msoa[f"{floor_type}_%"] = (
            infilled_msoa[f"msoa_{floor_type}"] / infilled_msoa["msoa_all"]
        )

    df = pd.merge(lsoa_with_msoa, infilled_msoa, how="left")

    for floor_type in FLOOR_TYPES:
        df[f"adj_{floor_type}_%"] = np.where(
            df[floor_type].isna(), df[f"{floor_type}_%"], 0
        )
    df["adj_all_%"] = df["adj_all_%"].fillna(0)

    df["adj_all_%"] = (
        df["adj_industry_%"]
        + df["adj_office_%"]
        + df["adj_other_%"]
        + df["adj_retail_%"]
    )

    return df


def infill_lsoa_floor_type_with_splits(df: pd.DataFrame) -> pd.DataFrame:
    for floor_type in FLOOR_TYPES:
        df[f"{floor_type}_infilled"] = df[floor_type]
        df[f"{floor_type}_infilled"] = df[f"{floor_type}_infilled"].fillna(0)

    df["total_from_parts"] = (
        df["industry_infilled"]
        + df["office_infilled"]
        + df["other_infilled"]
        + df["retail_infilled"]
    )

    # don't allow to_distribute to be negative
    df["to_distribute"] = df["all"] - df["total_from_parts"]
    df["to_distribute"] = df["to_distribute"].mask(df["to_distribute"] < 0, 0)

    for floor_type in FLOOR_TYPES:
        if floor_type != "all":
            df[f"adj_{floor_type}_%"] = df[f"adj_{floor_type}_%"] / df["adj_all_%"]
            # TODO change to being the missing amount not the total amount
            df[floor_type] = df[floor_type].fillna(
                df["to_distribute"] * df[f"adj_{floor_type}_%"]
            )

    return df


def attach_msoa_to_lsoa(df: pd.DataFrame) -> pd.DataFrame:
    lu = pd.read_csv(RAW_DIR / "ews_lsoa_to_msoa_lu.csv")
    lu.columns = ["ons_code", "msoa_code"]
    df = pd.merge(df, lu, how="left")
    return df


def calc_msoa_current_and_targets(
    lsoa_w_msoa: pd.DataFrame, infilled_msoa: pd.DataFrame
) -> pd.DataFrame:

    msoa_totals = lsoa_w_msoa.groupby("msoa_code")["all"].sum().reset_index()

    # reduce columns and rename to avoid clashes with lsoa table
    infilled_msoa = infilled_msoa[["ons_code", "all"]]
    infilled_msoa = infilled_msoa.rename(
        columns={"ons_code": "msoa_code", "all": "target_all"}
    )

    msoa_all_current_and_targets = pd.merge(msoa_totals, infilled_msoa, how="left")

    msoa_all_current_and_targets["diff"] = (
        msoa_all_current_and_targets["target_all"] - msoa_all_current_and_targets["all"]
    )

    msoa_all_current_and_targets.to_csv(
        INTERMIDIATE_DIR / "msoa_all_floorspace_current_and_targets.csv", index=False
    )

    return msoa_all_current_and_targets


def extract_file_data(filepath: Path, geography: str, measure: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, na_values=["..", "."])

    floor_type = get_floortype_from_filename(file=filepath)

    filtered = df[df["geography"] == geography].copy()

    filtered[measure] = floor_type

    return filtered


def msoa_balance_loop(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for floor_type in FLOOR_TYPES:
        df[f"{floor_type}_working"] = np.where(
            df[floor_type].isna(),
            df[f"{floor_type}_working"] * df["factor"],
            df[floor_type],
        )

    df["working_total"] = (
        df["industry_working"]
        + df["office_working"]
        + df["other_working"]
        + df["retail_working"]
    )

    df["to_be_allocated"] = df["all"] - df["working_total"]

    df["factor"] = df["all"] / df["working_total"]

    df["factor"] = df["factor"].fillna(1.0)
    return df


def get_floortype_from_filename(file: Path) -> str:
    file_stem_starts = file.stem[:-1]
    if file_stem_starts == "table_FS_OA1_":
        return "all"
    if file_stem_starts == "table_FS_OA2_":
        return "retail"
    if file_stem_starts == "table_FS_OA3_":
        return "office"
    if file_stem_starts == "table_FS_OA4_":
        return "industry"
    if file_stem_starts == "table_FS_OA5_":
        return "other"
    raise ValueError(f"Unknown floor type for filename: {file_stem_starts}")


def process_voa_rates() -> pd.DataFrame:
    # Note the _2 which are rates
    filepaths = list(RAW_DIR.glob("table_*_2.csv"))

    geographies = ["LAUA", "MSOA", "LSOA"]

    voa_values = {}

    for geography in geographies:
        dfs_wide = prepare_voa_value_data(filepaths=filepaths, geography=geography)
        # attach geography from above to each
        if geography == "MSOA":
            lu = pd.read_csv(RAW_DIR / "ews_msoa_to_lad_lu.csv")
            lu.columns = ["msoa_code", "lad_code"]
            dfs_wide = pd.merge(dfs_wide, lu)
        if geography == "LSOA":
            lu = pd.read_csv(RAW_DIR / "ews_lsoa_to_msoa_lu.csv")
            lu.columns = ["lsoa_code", "msoa_code"]
            dfs_wide = pd.merge(dfs_wide, lu)

        voa_values[geography] = dfs_wide

    msoa_with_lad = pd.merge(voa_values["MSOA"], voa_values["LAUA"])

    all_geos = pd.merge(voa_values["LSOA"], msoa_with_lad)
    all_geos.to_csv(INTERMIDIATE_DIR / "pre-infill_voa_rates.csv", index=False)

    for floor_type in FLOOR_TYPES:
        # infill msoa from lad
        # infill lsoa from msoa
        all_geos[f"{floor_type}_msoa"] = all_geos[f"{floor_type}_msoa"].fillna(
            all_geos[f"{floor_type}_lad"]
        )
        all_geos[f"{floor_type}_lsoa"] = all_geos[f"{floor_type}_lsoa"].fillna(
            all_geos[f"{floor_type}_msoa"]
        )

    all_geos.to_csv(INTERMIDIATE_DIR / "infilled_voa_rates.csv", index=False)

    return all_geos


def prepare_voa_value_data(filepaths: list[Path], geography: str) -> pd.DataFrame:
    df = extract_data_for_geography(filepaths=filepaths, geography=geography)
    df = df.drop(columns=["ons_name"])
    col_id = geography.lower().replace("laua", "lad")
    df = df.add_suffix(f"_{col_id}")
    df = df.rename(columns={f"ons_code_{col_id}": f"{col_id}_code"})
    return df


def calc_values(floorspaces: pd.DataFrame, rates: pd.DataFrame):
    """Multiply the floorspace by the rate to give a value

    Args:
        floorspace (pd.DataFrame): floorspace value by lsoa
        rates (pd.DataFrame): rate values by lsoa
    """
    floorspace_renamer = {}
    for floor_type in FLOOR_TYPES:
        floorspace_renamer[floor_type] = f"{floor_type}_floorspace"

    floorspaces = floorspaces.rename(columns=floorspace_renamer)

    rates_renamer = {}
    for floor_type in FLOOR_TYPES:
        rates_renamer[f"{floor_type}_lsoa"] = f"{floor_type}_rate"

    rates = rates.rename(columns=rates_renamer)

    voa_values = pd.merge(floorspaces, rates, how="left")

    for floor_type in FLOOR_TYPES:
        voa_values[f"{floor_type}_value"] = (
            voa_values[f"{floor_type}_floorspace"] * voa_values[f"{floor_type}_rate"]
        )

    output_cols = ["lsoa_code", "lsoa_name", "lad_code"]

    output_measure_cols = [
        col
        for col in voa_values.columns
        if col.endswith("_rate")
        or col.endswith("_floorspace")
        or col.endswith("_value")
    ]

    output_cols.extend(output_measure_cols)

    output_cols.remove("lsoas_missing_all_value")
    output_cols.remove("lsoa_all_infill_value")

    voa_values = voa_values[output_cols]

    voa_values.to_csv(
        INTERMIDIATE_DIR / "infilled_voa_floorspace_rate_and_values.csv", index=False
    )


if __name__ == "__main__":
    main()
