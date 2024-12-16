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
    # infilling values from geography above
    infilling_msoa_from_lad()
    infilling_lsoa_from_msoa()
    create_losa_voa_values()


def infilling_lsoa_from_msoa() -> None:
    """Infill lsoa using average values from msoa."""
    lsoa_input = pd.read_csv(INTERMIDIATE_DIR / "LSOA_voa_rate.csv")

    lsoa_input = lsoa_input.drop(columns="geography")
    lsoa_input = lsoa_input.rename(columns={"ons_name": "LSOA21NM"})
    lsoa_long = lsoa_input.melt(
        id_vars="LSOA21NM", var_name="floor_type", value_name="raw_value"
    )

    lsoa_msoa_lu = pd.read_csv(
        ONS_CORRESPONDENCE,
        usecols=["LSOA21NM", "MSOA21NM"],
    ).drop_duplicates()

    df = pd.merge(lsoa_long, lsoa_msoa_lu, how="left")

    df["MSOA21NM"] = np.where(
        df["MSOA21NM"].isna(), df["LSOA21NM"].str[:-1], df["MSOA21NM"]
    )

    df = find_masking_and_starting_value(df=df)

    df["input_value"] = df["input_value"].astype(float)

    average_values = (
        df.groupby(["MSOA21NM", "floor_type"])
        .agg(average_value=("input_value", "mean"))
        .reset_index()
    )

    df = pd.merge(df, average_values)

    df["input_value"] = df["input_value"].fillna(df["average_value"])

    df = df.drop(columns="average_value")

    df["current_value"] = df["input_value"]

    df.to_csv(INTERMIDIATE_DIR / "infilled_LSOA_voa_rate.csv", index=False)


def infilling_msoa_from_lad() -> None:
    """Infill msoa using a combination of lad values and furnessing."""

    msoa_input = pd.read_csv(INTERMIDIATE_DIR / "MSOA_voa_rate.csv")
    msoa_input = msoa_input.drop(columns="geography")
    msoa_input = msoa_input.rename(columns={"ons_name": "MSOA21NM"})
    msoa_long = msoa_input.melt(
        id_vars="MSOA21NM", var_name="floor_type", value_name="raw_value"
    )

    df = attach_lad_to_msoas(msoa_long)

    df = find_masking_and_starting_value(df=df)

    df["input_value"] = df["input_value"].astype(float)

    # NAs are automatically excluded by pandas
    average_values = (
        df.groupby(["LAD21NM", "floor_type"])
        .agg(average_value=("input_value", "mean"))
        .reset_index()
    )

    df = pd.merge(df, average_values)

    df["input_value"] = df["input_value"].fillna(df["average_value"])

    df = df.drop(columns="average_value")

    df["current_value"] = df["input_value"]

    # now onto balancing
    for _ in range(100):
        df = balance_msoa_values_df(df=df)

    df.to_csv(INTERMIDIATE_DIR / "infilled_MSOA_voa_rate.csv", index=False)


def attach_lad_to_msoas(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lad name column to data stored in msoa level.

    Args:
        df (pd.DataFrame): Data with msoa specified

    Returns:
        pd.DataFrame: Data with the addition of a lad name column
    """

    lad_msoa_lu = pd.read_csv(
        ONS_CORRESPONDENCE,
        usecols=["LAD21NM", "MSOA21NM"],
    ).drop_duplicates()

    # Add in missing data rows
    additional_data = pd.DataFrame(
        {"MSOA21NM": "Redbridge 041", "LAD21NM": "Redbridge"}, index=[0]
    )

    lad_msoa_lu = pd.concat([lad_msoa_lu, additional_data])

    return pd.merge(df, lad_msoa_lu)


def balance_msoa_values_df(df: pd.DataFrame) -> pd.DataFrame:
    """Balance the floortype values for the msoa level

    Args:
        df (pd.DataFrame): Data to be adjusted at msoa geography

    Returns:
        pd.DataFrame: Balanced to match floorspace totals at msoa level
    """

    # balance by floortype (for the msoa)

    area_value_floorspace_factors = calculate_msoa_value_floorspace_factors(df=df)

    df = pd.merge(df, area_value_floorspace_factors)

    df["current_value"] = df["current_value"] * df["factor"]

    # factor has been used
    df = df.drop(columns="factor")

    # balance within the msoa
    factors = calculate_msoa_values_factors(df=df)

    df = pd.merge(df, factors)

    df["current_value"] = df["current_value"] * df["factor"]

    # factor has been used
    df = df.drop(columns="factor")

    # limit change of non-masked values to be within a tollerance of provided value
    df = constrain_changes(df=df)
    return df


def calculate_msoa_values_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate factors at the floor_type level to reach msoa targets
    Factors are only applied where the absoluate difference is more than 0.5

    Args:
        df (pd.DataFrame): Data to be adjusted

    Returns:
        pd.DataFrame: Factored data
    """

    msoa_floorspace = pd.read_csv(INTERMIDIATE_DIR / "infilled_msoa_floorspace.csv")

    msoa_floorspace = msoa_floorspace.rename(columns={"ons_name": "MSOA21NM"})

    msoa_floorspace = msoa_floorspace.drop(columns="geography")

    msoa_floorspace_long = pd.melt(
        msoa_floorspace,
        id_vars=["MSOA21NM"],
        var_name="floor_type",
        value_name="msoa_floorspace",
    )

    df = pd.merge(df, msoa_floorspace_long)

    target_values = df[df["floor_type"] == "all"]
    target_values = target_values[["MSOA21NM", "current_value"]]

    sub_values = df[df["floor_type"] != "all"]

    sub_values = sub_values.copy()

    sub_values["fx"] = sub_values["current_value"] * sub_values["msoa_floorspace"]

    current_ave_values = (
        sub_values.groupby(["MSOA21NM"])
        .agg({"fx": "sum", "msoa_floorspace": "sum"})
        .reset_index()
    )

    current_ave_values["ave_value"] = (
        current_ave_values["fx"] / current_ave_values["msoa_floorspace"]
    )

    current_ave_w_targets = pd.merge(current_ave_values, target_values, how="left")

    current_ave_w_targets["diff"] = (
        current_ave_w_targets["current_value"] - current_ave_w_targets["ave_value"]
    ).abs()

    current_ave_w_targets["factor"] = (
        current_ave_w_targets["current_value"] / current_ave_w_targets["ave_value"]
    )

    # fix factors to 1 where the difference is less than 0.5
    current_ave_w_targets["factor"] = np.where(
        current_ave_w_targets["diff"] < 0.5, 1.0, current_ave_w_targets["factor"]
    )

    factors = current_ave_w_targets[["MSOA21NM", "factor"]]

    return factors


def calculate_msoa_value_floorspace_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate factors at the floor_type level to reach lad targets
    Factors are only applied where the absoluate difference is more than 0.5

    Args:
        df (pd.DataFrame): Data to be adjusted

    Returns:
        pd.DataFrame: Factored data
    """

    lad_targets = prepare_lad_for_merge()

    # read in floorspace values
    msoa_floorspace = pd.read_csv(INTERMIDIATE_DIR / "infilled_msoa_floorspace.csv")

    msoa_floorspace = msoa_floorspace.drop(columns="geography")

    msoa_floorspace = msoa_floorspace.rename(columns={"ons_name": "MSOA21NM"})

    msoa_floorspace_long = msoa_floorspace.melt(
        id_vars="MSOA21NM", var_name="floor_type", value_name="floorspace"
    )

    msoa_df = pd.merge(df, msoa_floorspace_long, how="left")

    msoa_df["fx"] = msoa_df["current_value"] * msoa_df["floorspace"]

    current_ave = (
        msoa_df.groupby(["LAD21NM", "floor_type"])
        .agg({"fx": "sum", "floorspace": "sum"})
        .reset_index()
    )
    current_ave["current_ave"] = current_ave["fx"] / current_ave["floorspace"]

    current_ave_w_targets = pd.merge(current_ave, lad_targets, how="left")

    current_ave_w_targets["diff"] = (
        current_ave_w_targets["lad_rate"] - current_ave_w_targets["current_ave"]
    ).abs()

    current_ave_w_targets["factor"] = (
        current_ave_w_targets["lad_rate"] / current_ave_w_targets["current_ave"]
    )

    # fix factors to 1 where the difference is less than 0.5
    current_ave_w_targets["factor"] = np.where(
        current_ave_w_targets["diff"] < 0.5, 1.0, current_ave_w_targets["factor"]
    )

    floorspace_factors = current_ave_w_targets[["LAD21NM", "floor_type", "factor"]]

    return floorspace_factors


def prepare_lad_for_merge() -> pd.DataFrame:
    """Fetch voa rates at lad level, apply some steps to make the join to msoas smoother.

    Returns:
        pd.DataFrame: pre-processed voa rates at lad level
    """
    lad_input = pd.read_csv(INTERMIDIATE_DIR / "LAD_voa_rate.csv")

    lad_input = lad_input.drop(columns="geography")
    lad_input = lad_input.rename(columns={"ons_name": "LAD21NM"})
    lad_targets = lad_input.melt(
        id_vars="LAD21NM", var_name="floor_type", value_name="lad_rate"
    )

    # rename lads to match other datsets

    # strip UA from names
    lad_targets["LAD21NM"] = lad_targets["LAD21NM"].str.replace(" UA", "")

    # keep only first English version of name for Welsh LADs
    lad_targets["LAD21NM"] = lad_targets["LAD21NM"].str.split(pat=" /").str[0]

    # For Bristol and Hull
    lad_targets["LAD21NM"] = lad_targets["LAD21NM"].str.replace(", City of", "")

    # For Herefordshire
    lad_targets["LAD21NM"] = lad_targets["LAD21NM"].str.replace(", County of", "")
    return lad_targets


def find_masking_and_starting_value(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust inputs where the values are masked to give a numeric starting value

    Args:
        df (pd.DataFrame): Masked data

    Returns:
        pd.DataFrame: Infilled data with starting values
    """

    # only need to take care of .., as . whilst technically absent we know is 0
    df["masked"] = np.where(df["raw_value"] == "..", True, False)

    df["input_value"] = df["raw_value"]

    # if ".." then between 1 and 4 business
    df["input_value"] = np.where(df["raw_value"] == "..", np.NaN, df["input_value"])

    # if "." then no businesses for that combinaton
    df["input_value"] = np.where(df["raw_value"] == ".", 0, df["input_value"])

    # set values that were given as 0 to be 0.25 as we know they must lie between 0 and 0.5
    # if no businesses then would be given as 0 (see above)
    df["input_value"] = np.where(df["raw_value"] == 0, 0.25, df["input_value"])

    return df


def extract_voa_values_for(
    geography_type: str, measure: str, year: str = "2023"
) -> None:
    """Extract data from VOA provided csvs and write to file.

    Args:
        geography_type (str): Geography to be extracted.
        measure (str): Data to be extracted (curently either floorspace or rate)
        year (str, optional): Year column to extract. Defaults to "2023". Note needs to be string.
    """
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

    wide.to_csv(INTERMIDIATE_DIR / f"{geography_type}_voa_{measure}.csv", index=False)


def process_file(file: Path, geography_type: str) -> pd.DataFrame:
    """From file extract geography that we are interested in.
    Note that 2023 is hard coded at the moment.

    Args:
        file (Path): Path to file to be read
        geography_type (str): Geography to be extracted

    Returns:
        pd.DataFrame: Dataframe containing data we need
    """

    df = pd.read_csv(file)

    if "LAD" in geography_type:
        df = df.query('geography == "LAUA"')
    else:
        df = df.query(f'geography == "{geography_type}"')

    # TODO: make 2023 a variable to be passed in.

    reduced = df[["geography", "ons_name", "2023"]]
    reduced = reduced.dropna().copy()
    file_desc = get_floortype_from_filename(file=file)
    reduced["measure"] = file_desc
    return reduced


def get_floortype_from_filename(file: Path) -> str:
    """Find the floortype contained within a given filepath based on the filepath start.

    Args:
        file (Path): Path to file that has been read in.

    Raises:
        ValueError: File stem not recognised, suggesting either the file is not expected or this function needs expanding.

    Returns:
        str: The floor type contained with the csv.
    """
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
    """Infill the msoa floorspace values where they are masked. 
    This is done by first working out the amount of the total not allocated.
    And this is shared equally between the masked categories.
    """

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
        INTERMIDIATE_DIR / f"infilled_msoa_floorspace.csv", index=False
    )


def initial_infill_lsoa_floorspace() -> None:
    """Infill the lsoa floorspace values where they are masked. 
    This is done by first working out the amount of the total not allocated.
    And this is shared equally between the masked categories.
    """
    msoa_infilled_targets = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_msoa_floorspace.csv",
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
    """Attach msoa name column to data stored in lsoa level.

    Args:
        df (pd.DataFrame): Data with lsoa specified

    Returns:
        pd.DataFrame: Data with the addition of a msoa name column
    """

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
    """Furness the lsoa floorspace values, with some prep first and the looping 21 times. Writes output to file."""
    lsoa_targets = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_floorspace_targets.csv"
    )

    lsoa_targets = attach_msoa(lsoa_targets)

    measure_targets = pd.read_csv(INTERMIDIATE_DIR / "infilled_msoa_floorspace.csv")
    measure_targets = measure_targets.drop(columns=["geography", "all"])
    measure_targets = measure_targets.rename(columns={"ons_name": "msoa"})
    measure_targets_long = pd.melt(
        measure_targets,
        id_vars=["msoa"],
        var_name="floor_type",
        value_name="floor_type_target",
    )

    input_df = pd.read_csv(INTERMIDIATE_DIR / "LSOA_voa_floorspace.csv", na_values="..")

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

    # infill with starting values where we have missing data (e.g., Oxford 020 which is a missing MSOA)
    lsoa_long["current_value"] = lsoa_long["current_value"].fillna(
        lsoa_long["input_value"]
    )

    lsoa_long["current_value"] = lsoa_long["current_value"].fillna(0)

    lsoa_long.to_csv(INTERMIDIATE_DIR / "infilled_lsoa_voa_floorspace.csv")


def prepare_mosa_for_furness(df: pd.DataFrame) -> pd.DataFrame:
    """Infill masked value in the msoa dataset

    Args:
        df (pd.DataFrame): Data with masked values

    Returns:
        pd.DataFrame: Infilled data
    """

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
    """Perform one loop of furnesing on the data,
    First it factoring to match floor type values,
    Then lsoa constraints,
    Finally process ensures that where values were given (i.e., not masked) they don't change by more than 0.5

    Args:
        lsoa_long (pd.DataFrame): Data to be balanced
        measure_targets_long (pd.DataFrame): Targets giving floortype values
        lsoa_targets (pd.DataFrame): Values at lsoa to be reached

    Returns:
        pd.DataFrame: Twice balanced and constrained dataset
    """
    # factor to reach floor_types values across msoa
    once_factored = factor_to_floor_types(lsoa_long, measure_targets_long)

    # factor to reach lsoa constraint
    twice_factored = factor_to_lsoa_targets(df=once_factored, lsoa_targets=lsoa_targets)

    # limit changes to +/- 0.5 where value was provided
    twice_factored = constrain_changes(twice_factored)
    return twice_factored


def factor_to_floor_types(
    lsoa_long: pd.DataFrame, measure_targets_long: pd.DataFrame
) -> pd.DataFrame:
    """Adjust the data to match the floorspace targets.

    Args:
        lsoa_long (pd.DataFrame): Data to be adjusted
        measure_targets_long (pd.DataFrame): Targets for floortype at the msoa level

    Returns:
        pd.DataFrame: Adjusted data
    """
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
    df = pd.merge(lsoa_long, df, on=["msoa", "floor_type"], how="left")

    df["current_value"] = df["current_value"] * df["floor_type_factor"]

    df = df.drop(columns=["floor_type_factor"])
    return df


def factor_to_lsoa_targets(
    df: pd.DataFrame, lsoa_targets: pd.DataFrame
) -> pd.DataFrame:
    """Adjust the data to match the lsoa targets.

    Args:
        df (pd.DataFrame): Data to be adjusted
        lsoa_targets (pd.DataFrame): Targets at lsoa level

    Returns:
        pd.DataFrame: Adjusted data
    """

    # factor to reach lsoa targets
    msoa_totals = (
        df.groupby(["msoa", "ons_name"]).agg({"current_value": "sum"}).reset_index()
    )

    adj_factors = pd.merge(msoa_totals, lsoa_targets)

    adj_factors["lsoa_factor"] = (
        adj_factors["lsoa_target"] / adj_factors["current_value"]
    )

    adj_factors["lsoa_factor"] = np.where(
        adj_factors["lsoa_factor"].isna(), 1, adj_factors["lsoa_factor"]
    )

    adj_factors = adj_factors[["msoa", "ons_name", "lsoa_factor"]]
    # attach lsoa factors
    twice_factored = pd.merge(df, adj_factors, how="left")
    twice_factored["current_value"] = (
        twice_factored["current_value"] * twice_factored["lsoa_factor"]
    )

    return twice_factored


def constrain_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure non-masked values do not change by more than 0.5.

    Args:
        df (pd.DataFrame): Data to be adjusted

    Returns:
        pd.DataFrame: Adjusted data
    """

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


def create_losa_voa_values() -> None:
    """Write lsoa rates and floorspace values into one csv."""

    lsoa_rates = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_voa_rate.csv",
        usecols=["LSOA21NM", "floor_type", "current_value"],
    )
    lsoa_floorspace = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_voa_floorspace.csv",
        usecols=["ons_name", "floor_type", "current_value"],
    )

    lsoa_rates = lsoa_rates.rename(
        columns={"LSOA21NM": "lsoa21nm", "current_value": "rate"}
    )
    lsoa_floorspace = lsoa_floorspace.rename(
        columns={"ons_name": "lsoa21nm", "current_value": "floorspace"}
    )

    lsoa_floorspace_all = (
        lsoa_floorspace.groupby("lsoa21nm")["floorspace"].sum().reset_index()
    )

    lsoa_floorspace_all["floor_type"] = "all"

    lsoa_floorspace = pd.concat([lsoa_floorspace, lsoa_floorspace_all])

    lsoa_value = pd.merge(lsoa_rates, lsoa_floorspace)

    lsoa_value["voa_value"] = lsoa_value["rate"] * lsoa_value["floorspace"]

    lsoa_value.to_csv(INTERMIDIATE_DIR / "infilled_lsoa_voa_value.csv", index=False)


if __name__ == "__main__":
    main()
