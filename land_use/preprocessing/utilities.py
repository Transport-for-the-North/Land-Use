import logging
from pathlib import Path
from os import PathLike
from typing import Literal, Tuple
from warnings import warn

import pandas as pd

from land_use.constants import _MODEL_PRICE_BASE


def find_header_line(
        file_path: PathLike, 
        header_string: str, 
        max_lines: int = 100
    ) -> Tuple[int, str]:
    """Identifies (0-indexed) row within a file that contains some header text.

    Most useful for files that have some kind of information text stored above
    column headers that indicate the start of the actual data.

    Raises RuntimeError if `header_start` cannot be found in the first `max_lines` lines of
    text or the entirety of the file, whichever comes first.

    Parameters
    ----------
    file_path : PathLike
        File to search through
    header_string : str
        Text within the "column header" line. Note this will be converted
          to lowercase within the function as the search is case-insensitive.
          This does not impact the output as it is not returned. 
    max_lines : int, optional
        maximum number of line to check for the header text. Defaults to 100.

    Returns
    -------
    Tuple[int, str]
        (0-indexed) number of the first line within the file that begins with
        `header_start`, and the first column entry that contains header_string 
        (with the case being unaltered).
    """

    if len(header_string) < 2:
        warn(
            f'A very short suggested header text ("{header_string}") has been provided'
            f' - a starting string of at least 2 characters is recommended.'
        )

    # Convert starting text to lowercase
    header_low = header_string.lower()

    with open(file_path, 'r') as input_file:
        for i, line in enumerate(input_file):
            if header_low in line.lower():
                # get the section of the string (assumed to be comma delimited because we're reading csv)
                # that contains header_low and pass that back to use in subsequent stuff
                list_of_strings = line.split(',')
                list_of_relevant_strings = [item for item in list_of_strings if header_low in item.lower()]
                if len(list_of_relevant_strings) > 1:
                    warn(
                        f'Multiple column names in this file contain the string "{header_string}", these are {list_of_relevant_strings}.'
                        f'By default this will return the first relevant column name: {list_of_relevant_strings[0]}.'
                        f'If this is picking up the wrong column, please refine your input string.'
                    )
                # Exit on first match
                return i, list_of_relevant_strings[0].replace('"', '').strip()

            if i >= max_lines:
                break

        # Determine whether we're here because we've finished the file or not
        try:
            # If we can get another line from the file, we've not reached the end
            next(input_file)
            extra_text = f' the first {max_lines} lines of '
        except StopIteration:
            # Otherwise, we've maxed out the file.
            extra_text = ' '

    # Raise the error
    raise RuntimeError(
        f'Unable to find "{header_string}" in{extra_text}{file_path}'
    )


def read_headered_csv(
        file_path: PathLike, 
        header_string: str, 
        **kwargs
    ) -> Tuple[pd.DataFrame, str]:
    """Reads a CSV (or similarly-delimited file), skipping metadata stored before actual data

    Parameters
    ----------
    file_path : PathLike
        path to the file to read
    header_string : str
        text within the "column header" line. Note this will be converted
        to lowercase by the function.
    kwargs : 
        any other options to pass to `pd.read_csv`

    Returns
    -------
    Tuple[pd.DataFrame, str]:
        tuple of the dataframe of the data found after the header row, with the 
        index of the header row and the first column entry that contains header_string.
    """

    logging.info(f'Reading in {file_path}')
    skip_rows, col_name = find_header_line(file_path, header_string)

    return pd.read_csv(file_path, skiprows=skip_rows, **kwargs).dropna(), col_name


def read_in_excel(file: Path, tab: str, names: list|None = None) -> pd.DataFrame:
    """Reads in a spreadsheet worksheet based on the tab name provided

    Parameters
    ----------
    file : Path
        Path to directory containing the spreadsheet
    tab : str
        Tab name within spreadsheet to read in as data. This tab name must exist 
        in the excel book otherwise this function will fail.
    names : list, optional
        names to assign to the columns of the resulting dataframe. Should be 
        used where existing names in the Excel workbook are undesired, with the
        assumption that names are contained in row 1 **only**. The length 
        of this list must match the number of expected columns. By default, None
        (i.e. do not rename columns)

    Returns
    -------
    pd.DataFrame
        Dataframe of the data, either with headers supplied by the data itself, 
        or by the names list provided.

    Raises
    ------
    FileNotFoundError
        if the provided file does not already exist.
    """

    logging.info(f'Reading in {file}')
    if not file.is_file():
        raise FileNotFoundError(f'{file} cannot be found')

    if names:
        return pd.read_excel(
        file, sheet_name=tab, engine='openpyxl', skiprows=1, names=names
    )
    
    return pd.read_excel(file, sheet_name=tab, engine='openpyxl', header=0)


def save_preprocessed_hdf(
        source_file_path: Path, 
        df: pd.DataFrame, 
        multiple_output_ref: str|None = None,
        key: str = 'df',
        mode: Literal['a', 'w', 'r+'] = 'w'
    ) -> None:
    """Save a dataframe to HDF5 format, in a "preprocessing" subfolder.

    The output file location will be a subfolder in the file_path location named 'preprocessing'
    and the file name will have the same name as the file_path file.

    This is done to help maintain the link between the original input file (e.g. csv or excel)
    and the converted output file.

    Parameters
    ----------
    source_file_path : Path
        File path to the input file that has been read in and converted into the 
        DVector-readable format by one of the parsing functions.

    df : pd.DataFrame
        Data to be saved in HDF5 format with the same name as file_path.

    multiple_output_ref : str, default None
        This is a reference to include if multiple outputs will be output from the same input file. Instead
        of the hdf being named exactly as the input file, it will be named as the {input_file}_{multiple_output_ref}.hdf

    key: str, default df
        Key to use within the output hdf file.

    mode: Literal['a', 'w', 'r+'], default w
        Mode to write to hdf. Default of w write to file, replacing existing contents 
    Returns
    -------

    """
    output_folder = source_file_path.parent / 'preprocessing'
    output_folder.mkdir(exist_ok=True)

    if multiple_output_ref:
        filename = f'{source_file_path.with_suffix("").name}_{multiple_output_ref}.hdf'
    else:
        filename = source_file_path.with_suffix('.hdf').name
        
    logging.info(f'Writing to {output_folder / filename}')
    df.to_hdf(output_folder / filename, key=key, mode=mode)


def pivot_to_dvector(
        data: pd.DataFrame,
        zoning_column: str,
        index_cols: list,
        value_column: str
    ) -> pd.DataFrame:
    """Function to pivot a long format dataframe into DVector format,
    where the column headers are the zone names and index is a segmentation definition.

    Parameters
    ----------
    data : pd.DataFrame
        Data you wish to pivot to DVector format. This is assumed to be in
        long format with columns of (at least)
        [zoning_column] + index_cols
    zoning_column : str
        Name of the column containing the zone names. This should be non-duplicated!
    index_cols :
        List of columns to define as the segmentation index in the DVector.
    value_column :
        Name of the column containing the actual values of the data.
    Returns
    -------
    pd.DataFrame
        Dataframe with column names as the zoning_column values, and index of
        index_cols, with values from value_col
    """
    # restrict to columns of interest
    # TODO put in checks for duplicates in the data
    dropped = data.loc[:, [zoning_column] + index_cols + [value_column]]

    # set the index value to be the index cols plus the zoning col and unstack
    reindexed = dropped.set_index(
        [zoning_column] + index_cols
    ).unstack(
        level=[zoning_column]
    )

    # set column names to be the zoning column values
    reindexed.columns = reindexed.columns.get_level_values(zoning_column)

    # TODO: consider if to enforce a type change of making all cells floats.
    # As this helps with HDF writing/compressing.

    return reindexed

def extract_geo_code(
    col: pd.Series, england: bool = True, wales: bool = True, scotland: bool = True, nireland: bool = True
) -> pd.Series:
    """Extract LSOA/DataZone/MSOA type from a pandas series. 
    Done by assuming code begin with a country prefix followed by 8 digits.
    Option within function to allow countries to be excluded from the match.

    Args:
        col (pd.Series): Column to match pattern against.
        england (bool, optional): If to include code belonging to England (starting with an E). Defaults to True.
        wales (bool, optional): If to include code belonging to Wales (starting with an W). Defaults to True.
        scotland (bool, optional): If to include code belonging to Scotland (starting with an S). Defaults to True.
        nireland (bool, optional): If to include code belonging to Northern Ireland (starting with an 9). Defaults to True.

    Raises:
        ValueError: If no countries have been selected. Likely to be an error.

    Returns:
        pd.Series: Series showing the extracted code, or NaN if not found for that index. Will be the same length as the input col.
    """
    include = ""

    if england:
        include += "E"
    if wales:
        include += "W"
    if scotland:
        include += "S"
    if nireland:
        include += "9"

    if include == "":
        raise ValueError(f"No countries selected.")

    return col.str.extract(rf"([{include}]\d{{8}})", expand=False)

def reformat_xsoa_sic_digits_to_dvector(
    df: pd.DataFrame,
    heading_col: str,
    segmentation: dict[int, str],
    seg_name: str,
    zoning: str,
) -> pd.DataFrame:
    """Convert an input dataframe with required segmentation and zoning into DVector format.

    Parameters
    ----------
        df : pd.DataFrame
            Data as read in from file
        heading_col : str
            Column name that includes the geography code with extra information
        segmentation : dict[int, str]
            Used for mapping the segmentation labels to index numbers (starting at 1)
        seg_name : str
            Name for the segmentation
        zoning : str
            The name of the column that is the geography code (and NOTHING else)

    Returns
    -------
    pd.DataFrame:
        Data in DVector format (sic segmentation as index and geographical areas as columns).
    """

    # Extract geocodes and filter to England+Wales
    df[zoning] = extract_geo_code(df[heading_col], nireland=False)
    df = df.dropna(subset=[zoning])
    df = df.drop(columns=[heading_col])

    # Turn into long format to allow segmentation index to be allocated
    df = df.melt(id_vars=[zoning])

    # define dictionary of segmentation mapping
    inv_seg = {v: k for k, v in segmentation.items()}

    # map the definitions used to define the segmentation
    df[seg_name] = df["variable"].map(inv_seg)
    df[seg_name] = df[seg_name].astype(int)
    df = df.drop(columns=["variable"])

    # convert to dvector format
    df_wide = pivot_to_dvector(
        data=df,
        zoning_column=zoning,
        index_cols=[seg_name],
        value_column="value",
    )

    df_wide = df_wide.astype(float)

    return df_wide

def read_headered_and_tailed_csv(
    file_path: PathLike, header_string: str, **kwargs
) -> Tuple[pd.DataFrame, str]:
    """Reads a CSV (or similarly-delimited file), skipping metadata stored before and after actual data.
    Works by elimiating rows where the header string column is NA.

    Parameters
    ----------
    file_path : PathLike
        path to the file to read
    header_string : str
        text within the "column header" line. Note this will be converted
        to lowercase by the function.
    kwargs :
        any other options to pass to `pd.read_csv`

    Returns
    -------
    Tuple[pd.DataFrame, str]:
        tuple of the dataframe of the data found after the header row and before the ending data, with the
        index of the header row and the first column entry that contains header_string.
    """
    skip_rows, col_name = find_header_line(file_path, header_string=header_string)

    df = pd.read_csv(file_path, skiprows=skip_rows, **kwargs)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(subset=[col_name])

    return df, col_name

def reformat_2021_lad_4digit(
    df: pd.DataFrame,
    lad_lu: pd.DataFrame,
    segmentation: dict[int, str],
    seg_col: str,
    seg_name: str,
    zoning: str,
) -> pd.DataFrame:
    """Convert an input dataframe with required segmentation and LAD zoning into DVector format.

    Parameters
    ----------
    df : pd.DataFrame
        Data as read in from file
    lad_lu : pd.DataFrame
        Correspondence between LAD geo code and name
    segmentation : dict[int, str]
        Used for mapping the segmentation labels to index numbers (starting at 1)
    seg_name : str
        Name for the segmentation
    zoning : str
        The name of the column that is the geography code (and NOTHING else)

    Returns
    -------
    pd.DataFrame:
        Data in DVector format (sic segmentation as index and geographical LAD areas as columns).
    """

    # define dictionary of segmentation mapping
    inv_seg = {v: k for k, v in segmentation.items()}

    # map the definitions used to define the segmentation
    df[seg_name] = df[seg_col].map(inv_seg)
    df[seg_name] = df[seg_name].astype(int)
    df = df.drop(columns=seg_col)

    join_col = "LAD21NM"

    df_long = df.melt(id_vars=[seg_name], var_name=join_col)

    if not join_col in lad_lu.columns:
        raise ValueError(
            f"{join_col} is not found in lookup columns, {lad_lu.columns}. Check inputs."
        )

    df_with_codes = pd.merge(df_long, lad_lu, how="left", on=["LAD21NM"])

    df_with_codes[zoning] = extract_geo_code(df_with_codes[zoning], nireland=False)
    df_with_codes = df_with_codes.dropna(subset=[zoning])

    df_wide = pivot_to_dvector(
        data=df_with_codes,
        zoning_column=zoning,
        index_cols=[seg_name],
        value_column="value",
    )

    # set to float to help hdf process
    df_wide = df_wide.astype(float)

    return df_wide

def reformat_ons_sic_soc_correspondence(df: pd.DataFrame) -> pd.DataFrame:
    """Change the ONS sic-soc correspondence to be in DVector format. Values are in exact numbers not splits.

    Args:
        df (pd.DataFrame): Data in long format

    Returns:
        pd.DataFrame: Data in wide format (with SIC_1 digit and soc_9 as index and Regions as columns)
    """
    df = df.rename(
        columns={
            "Regions Code": "RGN2021",
            "Occupation (current) (10 categories) Code": "soc_9",
            "Industry (current) (19 categories) Code": "industry",
        }
    )

    df = df.query("soc_9 >0 and industry > 0")

    # Have to map from industry to 1 digit sic. 
    # Note that 1digit sic is longer as more disagreegate with industry 18 being allocated to 18,19,20,21
    # It is okay to duplicate these as we are only using splits.
    industry_to_sic_1_digit = pd.DataFrame(
        data={
            "industry": [x + 1 for x in range(21)],
            "sic_1_digit": [x + 1 for x in range(21)],
        }
    )

    industry_to_sic_1_digit.loc[
        industry_to_sic_1_digit["industry"] > 18, "industry"
    ] = 18

    df_with_sic = pd.merge(df, industry_to_sic_1_digit, how="left", on="industry").drop(
        columns=["industry"]
    )

    # TODO: consider if to move these to a correspondence file instead?
    soc_9_to_soc_3 = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3}

    df_with_sic["soc"] = df_with_sic["soc_9"].map(soc_9_to_soc_3)

    df_with_sic = df_with_sic.groupby(["RGN2021", "soc", "sic_1_digit"]).agg(
        {"Observation": "sum"}
    )

    df_with_sic = df_with_sic.reset_index()

    df_wide = df_with_sic.pivot(
        index=["sic_1_digit", "soc"], columns="RGN2021", values="Observation"
    )

    return df_wide

def reformat_hse_lad_4_digit(
        file_path: Path, 
        segmentation: dict[int,str],
        seg_name: str,
    ) -> pd.DataFrame:
    """Convert an input dataframe with required segmentation and LAD zoning into DVector format.

    Parameters
    ----------
    df : pd.DataFrame
        Data as read in from file
    segmentation : dict[int, str]
        Used for mapping the segmentation labels to index numbers (starting at 1)
    seg_name : str
        Name for the segmentation

    Returns
    -------
    pd.DataFrame:
        Data in DVector format (sic segmentation as index and geographical LAD areas as columns).
    """
    # TODO: add docstring

    df = pd.read_csv(file_path)

    df = df.drop(columns="Unnamed: 0")

    df = df.set_index("LAD23CD")

    df = df.transpose()

    df = df.reset_index()

    df["sic_code"] = df["index"].str[3:]
    # left pad with 0's to make a fixed width of 4
    df["sic_code"] = df["sic_code"].str.zfill(4)

    df = df.sort_values("sic_code")

    # define dictionary of segmentation mapping
    # slighty complicated by the fact the segementation has the description e.g.
    # 0100 : DEFRA/Scottish Executive Agricultural Data
    # and we just want `0100`
    inv_seg = {v.split(":")[0].strip(): k for k, v in segmentation.items()}

    # map the definitions used to define the segmentation
    df[seg_name] = df["sic_code"].map(inv_seg)
    df[seg_name] = df[seg_name].astype(int)
    df = df.drop(columns=["index", "sic_code"])

    df = df.set_index("sic_4_digit")
    return df


def convert_price_base(
        data: pd.DataFrame,
        deflator: pd.DataFrame,
        index_column: str,
        year_column: str = 'surveyyear',
        income_column: str = 'hh_income'
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data: pd.DataFrame
        Output of reduce_classified_build(), a semi-default version
        of TfN's NTS processing. Must contain columns `year_column` and
        `income_column`
    deflator: pd.DataFrame
        Price deflating data with *at least* `year_column` and
        `index_column`.
    index_column: str
        Column name in `deflator` that represents the year-specific adjustment
        indices that will rebase prices to a given year
    year_column: str, default 'surveyyear_SOURCE'
        Column name in `deflator` *and* `data` that represents the year
    income_column: str, default 'hh_income_SOURCE'
        Column name in `data` that represents the annual household income
        (or some other price data to convert)

    Returns
    -------
    pd.DataFrame
        `data` with an additional column of
        f'{income_column.replace('_SOURCE', '')}_rebased`

    """
    # combine NTS data with price deflator data based on survey year
    data = pd.merge(data, deflator, on=year_column, how='left')

    new_column = income_column.replace('_SOURCE', '')
    # convert the nominal prices in NTS to the model base year of 2023 prices
    data[f'{new_column}_{_MODEL_PRICE_BASE}'] = (
        data[income_column] / data[index_column]
    )

    # replace negative values with -1 (this is a missing NTS value)
    data[f'{new_column}_{_MODEL_PRICE_BASE}'] = (
        data[f'{new_column}_{_MODEL_PRICE_BASE}'].where(data[income_column] > 0, -1)
    )

    data[f'{new_column}_{_MODEL_PRICE_BASE}'] = data[f'{new_column}_{_MODEL_PRICE_BASE}'].astype(int)

    return data


def infill_for_years(
    df: pd.DataFrame,
    forecast_years: list[int],
    extroplate_beyond_end: Literal["static", "trend", "forbid"],
    min_reasonable_year: int = 1000,
    max_reasonable_year: int = 3000,
) -> pd.DataFrame:
    """Infill a dataframe with additional years. 
    Data within the provided range of years is extrpolated between the closet provided points.
    Years before this range are not allowed and raise an error.
    Behaviour for years after this range depend on the value provided by `extroplate_beyond_end`.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with yearly information in columns.
    forecast_years: list[int]
        Years you wish to add to the dataframe using interpolation, extrapolation or static trends.
    extroplate_beyond_end: Literal['static', 'trend', 'forbid']
        What to do for forecast years that are beyond the final year.
        static - Use the values provided in the final provided year
        trend - Continue the trend given by the final provided two years
        forbid - Do not allow a forecast year beyond the final provided year
    min_reasonable_year: int, optional, default 1000
        Lowest year that is expected in the dataset. To avoid pulling out a column that happens to have numeric name.
    max_reasonable_year: int, optional, default 3000
        Highest year that is expected in the dataset. To avoid pulling out a column that happens to have numeric name.
  
    Raises
    -------
    ValueError: If the minimum found year in the df is before min_reasonable_year
    ValueError: If the maximum found year in the df is after max_reasonable_year
    ValueError: If only one year is found in the df. Unable to extroplate.
    NotImplementedError: Not able to forecast before the first year.
    ValueError: Some of the forecast years asked for are not numeric.
    ValueError: If extroplate_beyond_end=="forbid" AND some forecast years are beyond the final year.
    NotImplementedError: Unhandled combination, due to gap in logic.

    Returns
    -------
    pd.DataFrame
        As df but with extra years columns (given by forecast_years).
    """

    years_in_df = [int(yr) for yr in df.columns if str(yr).isnumeric()]
    years_in_df.sort()
    min_year = years_in_df[0]
    max_year = years_in_df[-1]
    second_largest_year = years_in_df[-2]  # as it is sorted

    ### Checking inputs make sense/are compatible
    # make sure infered years are reasonable and not happening to be another column with a numeric name
    error_suffix = "this seems unlikely, and suggests an input/code incompatibility, where the column isn't a year."
    if min_year < min_reasonable_year:
        raise ValueError(f"Lowest year provided is {min_year}, {error_suffix}")
    if max_year > max_reasonable_year:
        raise ValueError(f"Highest year provided is {max_year}, {error_suffix}")
    if len(years_in_df) < 2:
        raise ValueError(
            f"Unable to infill as only found {len(years_in_df)} year in provided df: {years_in_df}."
        )
    if min(forecast_years) < min_year:
        # before first year, raise error for now
        raise NotImplementedError(
            f"Unable to extrapolate backwards for {min(forecast_years)}, earliest year is {min_year}"
        )
    non_numeric_forecast_years = [y for y in forecast_years if not str(y).isnumeric()]
    if len(non_numeric_forecast_years):
        raise ValueError(f"The following requested forecasts year are not numbers {non_numeric_forecast_years}.")
    if extroplate_beyond_end == "forbid":
        forecast_years_beyond_final_provided = [y for y in forecast_years if y > max_year]
        if forecast_years_beyond_final_provided:
            raise ValueError(
                f"Some forecast years are beyond the final provided year {max_year},",
                f"this is not permitted given parameter {extroplate_beyond_end=}."
                f"{forecast_years}."
                 )

    # Can now move onto the infilling
    for year in forecast_years:
        if year in df.columns:
            # column already exists
            pass
        elif year > max_year and extroplate_beyond_end == "static":
            # take last provided years as the value
            df[year] = df[max_year]
        else:
            if year < max_year:
                year_a = max([y for y in years_in_df if y < year])
                year_b = min([y for y in years_in_df if y > year])
            elif extroplate_beyond_end == "trend":
                year_a = second_largest_year
                year_b = max_year
            else:
                raise NotImplementedError(f"Unable to process for {year=}.")
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            df[year] = perc_of_a * df[year_a] + perc_of_b * df[year_b]
    return df
