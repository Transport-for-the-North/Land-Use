from pathlib import Path
import pandas as pd
import logging

import land_use.preprocessing as pp


# Include the base year in here as we are pivoting from it
YEARS_TO_CALCULATE = [2023, 2028, 2033, 2038, 2043, 2048]

LOGGER = logging.getLogger(__name__)

LMS_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\Labour Market and Skills")

region_corr = pd.read_csv(
    Path(
        r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
        "GOR2021_CD_NM_EWS.csv",
    )
)


def pre_process_lms_sic():
    """
    Function to read in and pre-process the Labour Market & Skills dataset for SIC Industry Table 2
    Outputs growth factors for each forecast year, as hdfs
    """
    sic = []
    # Read in and format the LM&S data for each region
    for region in region_corr["RGN21NM"]:
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
    # TODO question, 2025 to 2035 or 2015 to 2035?
    # Get jobs into 1000s and aggregate any SIC values with multiple LM&S definitions, e.g. SIC 3
    sic_rgns = (
        sic_rgns[["LU_SIC_1_digit", "region", "2025", "2035"]]
        .groupby(by=["LU_SIC_1_digit", "region"], as_index=False)[["2025", "2035"]]
        .sum()
    )
    sic_rgns["2025"] = sic_rgns["2025"] * 1000
    sic_rgns["2035"] = sic_rgns["2035"] * 1000
    # TODO numbers under 10,000?

    # Calculate % growth for 1 year
    sic_rgns["%_growth_1yr"] = (
        (sic_rgns["2035"] - sic_rgns["2025"]) / sic_rgns["2025"]
    ) / 10
    sic_rgns = sic_rgns[["LU_SIC_1_digit", "region", "%_growth_1yr"]]

    # Add rows for SIC levels -1, 20, 21
    rgns = pd.DataFrame(sic_rgns["region"].drop_duplicates())
    lvls = []
    for lvl in [-1, 20, 21]:
        df = rgns.copy()
        df["LU_SIC_1_digit"] = lvl
        lvls.append(df)
    lvls_static = pd.concat(lvls)
    lvls_static["%_growth_1yr"] = 0
    sic_rgns = pd.concat([sic_rgns, lvls_static]).sort_values("LU_SIC_1_digit")

    # Output as hdf, ready to be read in as DVector
    f_years = {"2028": 5, "2033": 10, "2038": 15, "2043": 20, "2048": 25}
    for year in f_years.keys():
        f_year_control = sic_rgns.copy()
        f_year_control[year] = 1 + f_year_control["%_growth_1yr"] * f_years.get(year)
        f_year_control = f_year_control[["LU_SIC_1_digit", "region", year]].rename(
            columns={"LU_SIC_1_digit": "sic_1_digit"}
        )
        f_year_control = f_year_control.astype({"sic_1_digit": "int"})
        # Remap region back to codes
        f_year_control["region"] = f_year_control["region"].map(
            dict((x, y) for y, x in dict(sorted(region_corr.values.tolist())).items())
        )

        # Check for negatives
        count = f_year_control[year].lt(0).sum()
        if count > 0:
            LOGGER.error(
                f"Extrapolating the LM&S growth to {year} results in negative values"
            )

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=f_year_control,
            zoning_column="region",
            index_cols=["sic_1_digit"],
            value_column=year,
        )
        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR / r"LMS_SIC_Ind2\LMS_SIC_1_digit_Ind2.hdf",
            df=df_wide,
            multiple_output_ref=year,
        )


def pre_process_lms_soc():
    soc = []
    # Read in and format the LM&S data for each region
    for region in region_corr["RGN21NM"]:
        df = pd.read_csv(
            LMS_INPUT_DIR / rf"LMS_SOC\LMS_Occ_T1_{region}.csv",
            header=[0],
            skiprows=[1, 11, 12],
        )
        df["region"] = region
        soc.append(df)
    soc_rgns = pd.concat(soc)

    # Map LM&S industries to our segmentation
    lms_soc_corr = pd.read_csv(LMS_INPUT_DIR / r"LMS_SOC\LMS_SOC3_corr.csv")
    soc_rgns = pd.merge(
        soc_rgns,
        lms_soc_corr,
        left_on="Levels (000s)",
        right_on="Occupation",
        how="left",
    ).dropna()
    # TODO question, 2025 to 2035 or 2015 to 2035?
    soc_rgns = (
        soc_rgns[["SOC", "region", "2025", "2035"]]
        .groupby(by=["SOC", "region"], as_index=False)[["2025", "2035"]]
        .sum()
    )

    # Get jobs into 1000s
    soc_rgns["2025"] = soc_rgns["2025"] * 1000
    soc_rgns["2035"] = soc_rgns["2035"] * 1000

    # Calculate % growth for 1 year
    soc_rgns["%_growth_1yr"] = (
        (soc_rgns["2035"] - soc_rgns["2025"]) / soc_rgns["2025"]
    ) / 10
    soc_rgns = soc_rgns[["SOC", "region", "%_growth_1yr"]]

    # Add rows for SOC level 4
    rgns = pd.DataFrame(soc_rgns["region"].drop_duplicates())
    df = rgns.copy()
    df["SOC"] = 4
    df["%_growth_1yr"] = 0
    soc_rgns = pd.concat([soc_rgns, df])

    # Output as hdf, ready to be read in as DVector
    f_years = {"2028": 5, "2033": 10, "2038": 15, "2043": 20, "2048": 25}
    for year in f_years.keys():
        f_year_control = soc_rgns.copy()
        f_year_control[year] = 1 + f_year_control["%_growth_1yr"] * f_years.get(year)
        f_year_control = f_year_control[["SOC", "region", year]]
        f_year_control = f_year_control.astype({"SOC": "int"}).rename(
            columns={"SOC": "soc"}
        )
        # Remap region back to codes
        f_year_control["region"] = f_year_control["region"].map(
            dict((x, y) for y, x in dict(sorted(region_corr.values.tolist())).items())
        )

        # Check for negatives
        count = f_year_control[year].lt(0).sum()
        if count > 0:
            LOGGER.error(
                f"Extrapolating the LM&S growth to {year} results in negative values"
            )

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=f_year_control,
            zoning_column="region",
            index_cols=["soc"],
            value_column=year,
        )
        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR / r"LMS_SOC\LMS_SOC_Occ_T1.hdf",
            df=df_wide,
            multiple_output_ref=year,
        )
