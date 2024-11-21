from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import")
RAW_DIR = INPUT_DIR / "Employment Attraction Distributions" / "raw data"
INTERMIDIATE_DIR = (
    INPUT_DIR / "Employment Attraction Distributions" / "intermediate steps"
)
ONS_CORRESPONDENCE = (
    INPUT_DIR
    / "ONS"
    / "Correspondence_lists"
    / "EW_LSOA21_TO_MSOA21_TO_LAD21_TO_GOR21.csv"
)


def main():
    geographies = ["LSOA", "MSOA", "LAD"]
    for measure in ["floorspace", "rate"]:
        for geography in geographies:
            extract_voa_values_for(geography_type=geography, measure=measure)
    # current infill and furness process only makes sense with floorspace, as rate is an average not a summation
    initial_infill_msoa_floorspace()
    initial_infill_lsoa_floorspace()
    furness_floorspace()

    infilling_lsoa_voa_rates_from_above()

    calculate_lsoa_voa_value()


def infilling_lsoa_voa_rates_from_above():
    """
    Where masked unfill LSOA by the closest provided geography above.
    So first go to MSOA. And then to LAD.
    LAD is a complete dataset.
    Where "." is given then the value is 0.
    Where ".." is given then the value is masked.
    """
    lsoa_input = pd.read_csv(INTERMIDIATE_DIR / "lsoa_voa_rate.csv")

    lsoa_long = turn_voa_input_to_long(lsoa_input, "LSOA21NM")

    lsoa_msoa_lu = pd.read_csv(
        ONS_CORRESPONDENCE,
        usecols=["LSOA21NM", "MSOA21NM"],
    ).drop_duplicates()

    df = pd.merge(lsoa_long, lsoa_msoa_lu, how="left")

    df["MSOA21NM"] = np.where(
        df["MSOA21NM"].isna(), df["LSOA21NM"].str[:-1], df["MSOA21NM"]
    )

    msoa_input = pd.read_csv(INTERMIDIATE_DIR / "msoa_voa_rate.csv")

    msoa_long = turn_voa_input_to_long(msoa_input, "MSOA21NM")

    df = pd.merge(df, msoa_long, how="left")

    df = attach_lad_to_msoas(df)

    lad_long = get_lad_long()

    df = pd.merge(df, lad_long, how="left")

    df["infilled_msoa_voa_rate"] = np.where(
        df["MSOA21NM_raw_rate"] == "..", df["lad_rate"], df["MSOA21NM_raw_rate"]
    )
    df["infilled_msoa_voa_rate"] = np.where(
        df["MSOA21NM_raw_rate"] == ".", 0, df["infilled_msoa_voa_rate"]
    )

    df["infilled_lsoa_voa_rate"] = np.where(
        df["LSOA21NM_raw_rate"] == "..",
        df["infilled_msoa_voa_rate"],
        df["LSOA21NM_raw_rate"],
    )

    df["infilled_lsoa_voa_rate"] = np.where(
        df["LSOA21NM_raw_rate"] == ".", 0, df["infilled_lsoa_voa_rate"]
    )

    df.to_csv(INTERMIDIATE_DIR / "infilled_lsoa_voa_rate.csv", index=False)


def turn_voa_input_to_long(df: pd.DataFrame, geo_name: str) -> pd.DataFrame:
    df = df.drop(columns="geography")
    df = df.rename(columns={"ons_name": geo_name})
    df_long = df.melt(
        id_vars=geo_name, var_name="floor_type", value_name=f"{geo_name}_raw_rate"
    )
    return df_long


def attach_lad_to_msoas(df: pd.DataFrame) -> pd.DataFrame:
    lad_msoa_lu = pd.read_csv(
        ONS_CORRESPONDENCE,
        usecols=["LAD21NM", "MSOA21NM"],
    ).drop_duplicates()

    additional_data = pd.DataFrame(
        {"MSOA21NM": "Redbridge 041", "LAD21NM": "Redbridge"}, index=[0]
    )

    lad_msoa_lu = pd.concat([lad_msoa_lu, additional_data])

    df = pd.merge(df, lad_msoa_lu)
    return df


def get_lad_long() -> pd.DataFrame:
    lad_input = pd.read_csv(INTERMIDIATE_DIR / "lad_voa_rate.csv")

    lad_input = lad_input.drop(columns="geography")
    lad_input = lad_input.rename(columns={"ons_name": "LAD21NM"})
    lad_long = lad_input.melt(
        id_vars="LAD21NM", var_name="floor_type", value_name="lad_rate"
    )

    # rename lads to match other datsets

    # strip UA from names
    lad_long["LAD21NM"] = lad_long["LAD21NM"].str.replace(" UA", "")

    # keep only first English version of name for Welsh LADs
    lad_long["LAD21NM"] = lad_long["LAD21NM"].str.split(pat=" /").str[0]

    # For Bristol and Hull
    lad_long["LAD21NM"] = lad_long["LAD21NM"].str.replace(", City of", "")

    # For Herefordshire
    lad_long["LAD21NM"] = lad_long["LAD21NM"].str.replace(", County of", "")
    return lad_long


def extract_voa_values_for(
    geography_type: str, measure: str, year: str = "2023"
) -> None:
    if measure == "floorspace":
        voa_tables = RAW_DIR.glob("table_*1.csv")
    elif measure == "rate":
        voa_tables = RAW_DIR.glob("table_*2.csv")

    dfs_long = []

    for file in voa_tables:
        df = process_file(file=file, geography_type=geography_type)
        dfs_long.append(df)

    df_long = pd.concat(dfs_long)

    wide = df_long.pivot(
        index=["geography", "ons_name"], columns="measure", values=year
    )

    wide = wide.sort_values(["ons_name"])

    wide = wide.reset_index()

    # lsoa needs a correction for one lsoa.
    if geography_type == "LSOA":
        wide["ons_name"] = np.where(
            wide["ons_name"] == "Swansea 025J", "Swansea 025F", wide["ons_name"]
        )

    wide.to_csv(
        INTERMIDIATE_DIR / f"{geography_type.lower()}_voa_{measure}.csv", index=False
    )


def process_file(file: Path, geography_type: str) -> pd.DataFrame:
    df = pd.read_csv(file)

    if "LAD" in geography_type:
        df = df.query('geography == "LAUA"')
    else:
        df = df.query(f'geography == "{geography_type}"')

    reduced = df[["geography", "ons_name", "2023"]]
    reduced = reduced.dropna().copy()
    file_desc = get_floortype_from_filename(file=file)
    reduced["measure"] = file_desc
    return reduced


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


def initial_infill_msoa_floorspace() -> None:

    input_df = pd.read_csv(
        INTERMIDIATE_DIR / f"MSOA_voa_floorspace.csv", na_values=".."
    )

    wide = input_df.copy()
    wide = wide.drop(columns=["geography"])
    long = wide.melt(id_vars=["ons_name"], var_name="floor_type", value_name="size")

    all_floor = long[long["floor_type"] == "all"].copy()

    type_floor = long[long["floor_type"] != "all"].copy()
    type_floor_no_missing = type_floor.dropna().copy()
    type_floor_no_missing["entered_values"] = 1

    grouped_types = (
        type_floor_no_missing.groupby(["ons_name"])
        .agg({"size": "sum", "entered_values": "count"})
        .reset_index()
    )

    grouped_types = grouped_types.rename(columns={"size": "sum_of_parts"})

    infilled_values = pd.merge(all_floor, grouped_types, how="outer", on="ons_name")

    infilled_values["missing_value"] = (
        infilled_values["size"] - infilled_values["sum_of_parts"]
    )

    infilled_values["infill_with"] = np.where(
        infilled_values["missing_value"] < 0,
        0,
        np.where(
            infilled_values["entered_values"] == 4,
            0,
            infilled_values["missing_value"] / (4 - infilled_values["entered_values"]),
        ),
    )
    # special case here if no values are provided which happens for Central Bedfordshire 023,
    # where we have to spread out the total size equally (as nothing better to use)
    infilled_values["infill_with"] = np.where(
        infilled_values["sum_of_parts"].isna(),
        infilled_values["size"] / 4,
        infilled_values["infill_with"],
    )

    infilled_values["factors"] = np.where(
        infilled_values["entered_values"] < 4,
        1,
        infilled_values["size"] / infilled_values["sum_of_parts"],
    )

    infilled_values = infilled_values.fillna(0)
    infilled_values = infilled_values[["ons_name", "infill_with", "factors"]]

    infilled_floor_types = pd.merge(type_floor, infilled_values)
    infilled_floor_types["infilled_value"] = np.where(
        infilled_floor_types["size"].isna(),
        infilled_floor_types["infill_with"],
        infilled_floor_types["size"] * infilled_floor_types["factors"],
    )

    infilled_floor_types_wide = infilled_floor_types.pivot(
        index=["ons_name"], columns=["floor_type"], values="infilled_value"
    ).reset_index()

    infilled_floor_types_wide = infilled_floor_types_wide[
        ["ons_name", "industry", "office", "other", "retail"]
    ]

    floor_type_all_df = input_df[["geography", "ons_name", "all"]]
    infilled_msoa = pd.merge(floor_type_all_df, infilled_floor_types_wide)
    infilled_msoa["all"] = infilled_msoa["all"].fillna(0)

    infilled_msoa.to_csv(
        INTERMIDIATE_DIR / f"infilled_msoa_voa_floorspace.csv", index=False
    )


def initial_infill_lsoa_floorspace() -> None:

    msoa_infilled_targets = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_msoa_voa_floorspace.csv",
        usecols=["ons_name", "all"],
    )
    msoa_infilled_targets = msoa_infilled_targets.rename(columns={"ons_name": "msoa"})

    lsoa_values = pd.read_csv(
        INTERMIDIATE_DIR / "LSOA_voa_floorspace.csv",
        usecols=["ons_name", "all"],
        na_values="..",
    )
    lsoa_values = attach_msoa(lsoa_values)

    lsoas_in_msoa = lsoa_values.copy()
    lsoas_in_msoa["lsoas_in_msoa"] = 1
    lsoas_in_msoa = (
        lsoas_in_msoa.groupby(["msoa"]).agg({"lsoas_in_msoa": "count"}).reset_index()
    )

    lsoa_values_non_missing = lsoa_values.dropna().copy()
    lsoa_values_non_missing["entered_values"] = 1

    msoa_totals = (
        lsoa_values_non_missing.groupby(["msoa"])
        .agg({"all": "sum", "entered_values": "count"})
        .reset_index()
    )

    msoa_totals = msoa_totals.rename(columns={"all": "sum_of_parts"})

    lsoa_w_msoa_totals = pd.merge(msoa_totals, lsoas_in_msoa)

    lsoa_w_msoa_totals["na_values"] = (
        lsoa_w_msoa_totals["lsoas_in_msoa"] - lsoa_w_msoa_totals["entered_values"]
    )

    lsoa_with_adjustments = pd.merge(lsoa_w_msoa_totals, msoa_infilled_targets)
    lsoa_with_adjustments["missing_value"] = (
        lsoa_with_adjustments["all"] - lsoa_with_adjustments["sum_of_parts"]
    )

    lsoa_with_adjustments["infill_with"] = np.where(
        lsoa_with_adjustments["missing_value"] < 0,
        0,
        np.where(
            lsoa_with_adjustments["na_values"] == 0,
            0,
            lsoa_with_adjustments["missing_value"] / lsoa_with_adjustments["na_values"],
        ),
    )

    lsoa_with_adjustments["factors"] = np.where(
        lsoa_with_adjustments["na_values"] > 0,
        1,
        lsoa_with_adjustments["all"] / lsoa_with_adjustments["sum_of_parts"],
    )
    lsoa_with_adjustments = lsoa_with_adjustments.fillna(0)
    lsoa_with_adjustments = lsoa_with_adjustments[["msoa", "infill_with", "factors"]]

    lsoa_targets = pd.merge(lsoa_values, lsoa_with_adjustments)
    lsoa_targets["lsoa_target"] = np.where(
        lsoa_targets["all"].isna(),
        lsoa_targets["infill_with"],
        lsoa_targets["all"] * lsoa_targets["factors"],
    )
    lsoa_targets = lsoa_targets[["ons_name", "lsoa_target"]]
    lsoa_targets.to_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_floorspace_targets.csv", index=False
    )


def attach_msoa(df: pd.DataFrame) -> pd.DataFrame:
    lsoa_msoa_lu = pd.read_csv(
        ONS_CORRESPONDENCE,
        usecols=["LSOA21NM", "MSOA21NM"],
    )

    lsoa_msoa_lu = lsoa_msoa_lu.rename(
        columns={"LSOA21NM": "ons_name", "MSOA21NM": "msoa"}
    )
    df = pd.merge(df, lsoa_msoa_lu)
    return df


def furness_floorspace():
    lsoa_targets = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_floorspace_targets.csv"
    )

    lsoa_targets = attach_msoa(lsoa_targets)

    measure_targets = pd.read_csv(INTERMIDIATE_DIR / "infilled_msoa_voa_floorspace.csv")
    measure_targets = measure_targets.drop(columns=["geography", "all"])
    measure_targets = measure_targets.rename(columns={"ons_name": "msoa"})
    measure_targets_long = pd.melt(
        measure_targets,
        id_vars=["msoa"],
        var_name="floor_type",
        value_name="floor_type_target",
    )

    input_df = pd.read_csv(INTERMIDIATE_DIR / "lsoa_voa_floorspace.csv", na_values="..")

    input_df = input_df.drop(columns=["geography", "all"])

    lsoa_long = pd.melt(
        input_df, id_vars=["ons_name"], var_name="floor_type", value_name="input_value"
    )

    lsoa_long = attach_msoa(lsoa_long)

    lsoa_long["masked"] = np.where(
        np.isnan(lsoa_long["input_value"].values), True, False
    )

    lsoa_long = prepare_mosa_for_furness(df=lsoa_long)

    entries_pre_furness = len(lsoa_long)

    # 21 loops seems enough, could perhaps experiment with smaller number to see impact
    for _ in range(21):
        lsoa_long = furness_loop(
            lsoa_long=lsoa_long,
            measure_targets_long=measure_targets_long,
            lsoa_targets=lsoa_targets,
        )
        # make sure we haven't lost any rows
        assert len(lsoa_long) == entries_pre_furness

    # infill with starting values where we have missing data (e.g., Oxford 020 which is a missing msoa)
    lsoa_long["current_value"] = lsoa_long["current_value"].fillna(
        lsoa_long["input_value"]
    )

    lsoa_long["current_value"] = lsoa_long["current_value"].fillna(0)

    totals = (
        lsoa_long.groupby(["ons_name", "msoa"])
        .agg({"input_value": "sum", "current_value": "sum"})
        .reset_index()
    )

    # put in some defaults
    # TODO: consider if there is a better way to do this avoiding having to use np.NaN, as won't be the case for all rows
    totals["floor_type"] = "all"
    totals["masked"] = np.NaN
    totals["lsoa_factor"] = np.NaN

    lsoa_long = pd.concat([lsoa_long, totals])

    lsoa_long.to_csv(INTERMIDIATE_DIR / "infilled_lsoa_voa_floorspace.csv", index=False)


def prepare_mosa_for_furness(df: pd.DataFrame) -> pd.DataFrame:

    # NAs needs to be converted to a seed where na (which may be 0.1, or average of something)

    msoa_averages = (
        df.groupby(["msoa", "floor_type"]).agg({"input_value": "mean"}).reset_index()
    )

    msoa_averages = msoa_averages.rename(columns={"input_value": "average_input"})

    df_with_averages = pd.merge(df, msoa_averages)

    df_with_averages["current_value"] = np.where(
        df_with_averages["input_value"].isna(),
        df_with_averages["average_input"],
        df_with_averages["input_value"],
    )
    df_with_averages["current_value"] = np.where(
        df_with_averages["current_value"].isna(), 0, df_with_averages["current_value"]
    )

    # we know that 0's must be less than 0.5. So we set the seed as being just less than 0.5.
    # set 0 value to be 0.45 to allow these to grow/shrink as required.
    df_with_averages["current_value"] = np.where(
        df_with_averages["current_value"] == 0, 0.45, df_with_averages["current_value"]
    )

    df_with_averages = df_with_averages.drop(columns="average_input")

    return df_with_averages


def furness_loop(
    lsoa_long: pd.DataFrame,
    measure_targets_long: pd.DataFrame,
    lsoa_targets: pd.DataFrame,
) -> pd.DataFrame:
    # factor to reach floor_types values across msoa
    df2 = factor_to_floor_types(lsoa_long, measure_targets_long)

    # factor to reach lsoa constraint
    twice_factored = factor_to_lsoa_targets(df=df2, lsoa_targets=lsoa_targets)

    # limit changes to +/- 0.5 where value was provided
    twice_factored = constrain_changes(twice_factored)
    return twice_factored


def factor_to_floor_types(
    lsoa_long: pd.DataFrame, measure_targets_long: pd.DataFrame
) -> pd.DataFrame:
    current_floor_type_totals = (
        lsoa_long.groupby(["msoa", "floor_type"])
        .agg({"current_value": "sum"})
        .reset_index()
    )

    df = pd.merge(current_floor_type_totals, measure_targets_long)

    df["floor_type_factor"] = df["floor_type_target"] / df["current_value"]

    df["floor_type_factor"] = np.where(
        df["floor_type_factor"].isna(), 1, df["floor_type_factor"]
    )

    df = df[["msoa", "floor_type", "floor_type_factor"]]

    # attach floor type factors
    df2 = pd.merge(lsoa_long, df, on=["msoa", "floor_type"], how="left")

    df2["current_value"] = df2["current_value"] * df2["floor_type_factor"]

    df2 = df2.drop(columns=["floor_type_factor"])
    return df2


def factor_to_lsoa_targets(
    df: pd.DataFrame, lsoa_targets: pd.DataFrame
) -> pd.DataFrame:
    # factor to reach lsoa targets
    df3 = df.groupby(["msoa", "ons_name"]).agg({"current_value": "sum"}).reset_index()

    df4 = pd.merge(df3, lsoa_targets)

    df4["lsoa_factor"] = df4["lsoa_target"] / df4["current_value"]

    df4["lsoa_factor"] = np.where(df4["lsoa_factor"].isna(), 1, df4["lsoa_factor"])

    df4 = df4[["msoa", "ons_name", "lsoa_factor"]]
    # attach lsoa factors
    twice_factored = pd.merge(df, df4, how="left")
    twice_factored["current_value"] = (
        twice_factored["current_value"] * twice_factored["lsoa_factor"]
    )

    return twice_factored


def constrain_changes(df: pd.DataFrame) -> pd.DataFrame:
    df["change_from_start"] = np.where(
        df["masked"],
        0,
        df["current_value"] - df["input_value"],
    )

    # make change max of 0.5
    df["current_value"] = np.where(
        df["change_from_start"] > 0.5,
        df["input_value"] + 0.5,
        np.where(
            df["change_from_start"] < -0.5,
            df["input_value"] - 0.5,
            df["current_value"],
        ),
    )

    # remove extra column that we do not need
    df = df.drop(columns="change_from_start")

    return df


def calculate_lsoa_voa_value():

    lsoa_floorspace = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_voa_floorspace.csv",
        usecols=["ons_name", "floor_type", "current_value"],
    )
    lsoa_rates = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_voa_rate.csv",
        usecols=["LSOA21NM", "LAD21NM", "floor_type", "infilled_lsoa_voa_rate"],
    )

    lsoa_floorspace = lsoa_floorspace.rename(columns={"ons_name": "lsoa21nm"})

    lsoa_rates.columns = map(str.lower, lsoa_rates.columns)

    df = pd.merge(lsoa_floorspace, lsoa_rates, how="outer")

    df.to_csv(INTERMIDIATE_DIR / "infilled_lsoa_voa_value.csv", index=False)


if __name__ == "__main__":
    main()
