from pathlib import Path
import pandas as pd
import logging

import land_use.preprocessing as pp


# Include the base year in here as we are pivoting from it
BASE_YEAR = 2023
YEARS_TO_CALCULATE = [2023, 2028, 2033, 2038, 2043, 2048, 2053]

LOGGER = logging.getLogger(__name__)

LMS_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\Labour Market and Skills")

REGION_CORRESPONDENCE = pd.read_csv(
    Path(
        r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
        "GOR2021_CD_NM_EWS.csv",
    )
)


def main():
    pre_process_lms_sic()
    pre_process_lms_soc_by_g()


def pre_process_lms_sic() -> None:
    """
    Function to read in and pre-process the Labour Market & Skills dataset for SIC Industry Table 2
    Outputs totals for each year, based on LM&S growths
    """
    sic = []
    # Read in and format the LM&S data for each region
    for region in REGION_CORRESPONDENCE["RGN21NM"]:
        df = pd.read_csv(
            LMS_INPUT_DIR / rf"LMS_SIC_Ind2\LMS_Ind2_{region}.csv",
            header=[0],
            skiprows=[1],
        ).dropna()
        df = df.rename(columns={df.columns[0]: "Industry"})
        df["region"] = region
        sic.append(df)
    sic_rgns = pd.concat(sic)

    # Map LM&S industries to our segmentation
    lms_sic_corr = pd.read_csv(
        LMS_INPUT_DIR / r"LMS_SIC_Ind2\LMS_SIC_1_digit_corr.csv",
        dtype={"LU_SIC_1_digit": int},
    )
    sic_rgns = pd.merge(
        sic_rgns,
        lms_sic_corr,
        left_on="Industry",
        right_on="Labour Market & Skills",
        how="left",
    ).dropna()

    # Find years in df
    years = [int(yr) for yr in sic_rgns.columns if yr.isnumeric()]

    # Convert years columns names into integers
    column_mapper = {}
    for y in years:
        column_mapper[str(y)] = int(y)
    sic_rgns = sic_rgns.rename(columns=column_mapper)

    # Get jobs into 1000s and aggregate any SIC values with multiple LM&S definitions, e.g. SIC 3
    sic_rgns[years] = sic_rgns[years] * 1000
    sic_rgns = (
        sic_rgns[["LU_SIC_1_digit", "region"] + years]
        .groupby(by=["LU_SIC_1_digit", "region"], as_index=False)[years]
        .sum()
    )
    # TODO numbers under 10,000?

    # Interpolate for the years we want
    max_year = max(years)
    min_year = min(years)

    for year in YEARS_TO_CALCULATE:
        print(year)
        if year in sic_rgns.columns:
            # column already exists so nothing to do
            pass
        elif year < min_year:
            # before first year, raise error for now
            raise ValueError(
                f"Unable to extrapolate for {year}, earliest year is {min_year}"
            )
        elif year > max_year:
            # As 2035 is the maximum LM&S year, continue the 2025 to 2035 trend
            year_a = 2025
            year_b = 2035
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            sic_rgns[year] = perc_of_a * sic_rgns[year_a] + perc_of_b * sic_rgns[year_b]
        else:
            year_a = max([y for y in years if y < year])
            year_b = min([y for y in years if y > year])
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            sic_rgns[year] = perc_of_a * sic_rgns[year_a] + perc_of_b * sic_rgns[year_b]

    # Prepare for export
    sic_rgns = sic_rgns.rename(columns={"LU_SIC_1_digit": "sic_1_digit"}).astype(
        {"sic_1_digit": int}
    )

    # Remap region back to codes
    sic_rgns["region"] = sic_rgns["region"].map(
        dict(
            (x, y)
            for y, x in dict(sorted(REGION_CORRESPONDENCE.values.tolist())).items()
        )
    )

    # Output as hdf, ready to be read in as DVector
    for year in YEARS_TO_CALCULATE:
        sic_output = sic_rgns[["sic_1_digit", "region", year]]
        # Check for negatives
        count = sic_output[year].lt(0).sum()
        if count > 0:
            LOGGER.error(
                f"Extrapolating the LM&S growth to {year} results in negative values"
            )

        # Add rows for SIC levels -1, 20, 21
        rgns = pd.DataFrame(sic_rgns["region"].drop_duplicates())
        lvls = []
        for lvl in [-1, 20, 21]:
            df = rgns.copy()
            df["sic_1_digit"] = lvl
            lvls.append(df)
        lvls_static = pd.concat(lvls)
        lvls_static[year] = 0
        sic_output = pd.concat([sic_output, lvls_static]).sort_values("sic_1_digit")

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=sic_output,
            zoning_column="region",
            index_cols=["sic_1_digit"],
            value_column=year,  # function expects a string but int works here as matches column heading type
        )
        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR / r"LMS_SIC_Ind2\LMS_SIC_1_digit_Ind2.hdf",
            df=df_wide,
            multiple_output_ref=str(year),
        )


def pre_process_lms_soc_by_g() -> None:
    soc = []
    # Read in and format the LM&S data for each region
    for region in REGION_CORRESPONDENCE["RGN21NM"]:
        for g in ["Males", "Females"]:
            df = pd.read_csv(
                LMS_INPUT_DIR / rf"LMS_SOC\LMS_Occ_T1_{g}_{region}.csv",
                header=[0],
                skiprows=[1, 11, 12],
            )
            df["region"] = region
            df["g"] = g
            soc.append(df)
    soc_rgns = pd.concat(soc)

    # Map LM&S industries to our segmentation
    lms_soc_corr = pd.read_csv(LMS_INPUT_DIR / r"LMS_SOC\LMS_SOC3_corr.csv")
    soc_rgns = pd.merge(
        soc_rgns,
        lms_soc_corr,
        left_on=["Levels (000s)"],
        right_on=["Occupation"],
        how="left",
    ).dropna()

    # Find years in df
    years = [int(yr) for yr in soc_rgns.columns if yr.isnumeric()]

    # convert years columns names into integers
    column_mapper = {}
    for y in years:
        column_mapper[str(y)] = int(y)

    soc_rgns = soc_rgns.rename(columns=column_mapper)

    # Get jobs into 1000s
    soc_rgns[years] = soc_rgns[years] * 1000

    # now need to work on interpolating for the years we want
    max_year = max(years)
    min_year = min(years)

    for year in YEARS_TO_CALCULATE:
        print(year)
        if year in soc_rgns.columns:
            # column already exists so nothing to do
            pass
        elif year > max_year:
            # beyond year end so for soc we just take the last year as using it for proportions
            soc_rgns[year] = soc_rgns[max_year]
        elif year < min_year:
            # before first year, raise error for now
            raise ValueError(
                f"Unable to extrapolate for {year}, earliest year is {min_year}"
            )
        else:
            year_a = max([y for y in years if y < year])
            year_b = min([y for y in years if y > year])
            year_gap = year_b - year_a
            perc_of_a = 1 - ((year - year_a) / year_gap)
            perc_of_b = 1 - perc_of_a
            soc_rgns[year] = perc_of_a * soc_rgns[year_a] + perc_of_b * soc_rgns[year_b]

    # prepare for export
    soc_rgns = soc_rgns.astype({"SOC": "int"}).rename(columns={"SOC": "soc"})

    soc_rgns["g"] = soc_rgns["g"].map({"Males": 1, "Females": 2})

    # Remap region back to codes
    soc_rgns["region"] = soc_rgns["region"].map(
        dict(
            (x, y)
            for y, x in dict(sorted(REGION_CORRESPONDENCE.values.tolist())).items()
        )
    )

    soc_rgns = soc_rgns.groupby(["soc", "region", "g"]).sum().reset_index()

    soc_rgns[f"soc_prop_{BASE_YEAR}"] = soc_rgns[BASE_YEAR] / (
        soc_rgns.groupby(["region", "g"])[BASE_YEAR].transform("sum")
    )

    # Output as hdf, ready to be read in as DVector
    for year in YEARS_TO_CALCULATE:

        soc_rgns[f"soc_prop_{year}"] = soc_rgns[year] / (
            soc_rgns.groupby(["region", "g"])[year].transform("sum")
        )

        soc_rgns[f"soc_pp_change_from_{BASE_YEAR}_to_{year}"] = (
            soc_rgns[f"soc_prop_{year}"] - soc_rgns[f"soc_prop_{BASE_YEAR}"]
        )

        # Check for negatives
        count = soc_rgns[year].lt(0).sum()
        if count > 0:
            LOGGER.error(
                f"Extrapolating the LM&S growth to {year} results in negative values"
            )

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=soc_rgns,
            zoning_column="region",
            index_cols=["g", "soc"],
            value_column=f"soc_pp_change_from_{BASE_YEAR}_to_{year}",
        )
        out_stem = f"soc_pp_change_from_{BASE_YEAR}.hdf"
        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR / "LMS_SOC" / out_stem,
            df=df_wide,
            key=f"from_{BASE_YEAR}_to_{year}",
            mode="a"
        )


if __name__ == "__main__":
    main()
