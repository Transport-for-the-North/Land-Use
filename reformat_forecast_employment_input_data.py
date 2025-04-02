from pathlib import Path
import pandas as pd
import logging

import land_use.preprocessing as pp


# Include the base year in here as we are pivoting from it
BASE_YEAR = 2023
FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048, 2053]

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
    pre_process_lms_soc_by_g(separate_by_g=True)


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

    sic_rgns = pp.infill_for_years(
        df=sic_rgns, forecast_years=FORECAST_YEARS, extroplate_beyond_end="trend"
    )

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
    for year in FORECAST_YEARS:
        sic_output = sic_rgns
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

        sic_output[f"factor_from_{BASE_YEAR}_to_{year}"] = (
            sic_output[year] / sic_output[BASE_YEAR]
        )

        # Into a wide format for DVector
        df_wide = pp.pivot_to_dvector(
            data=sic_output,
            zoning_column="region",
            index_cols=["sic_1_digit"],
            value_column=f"factor_from_{BASE_YEAR}_to_{year}",
        )

        # infill the nas with 1 (this will be dividing by zero, coming from the static sic levels -1, 20, 21)
        print("infilled with 1's")
        df_wide = df_wide.fillna(1)

        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR
            / "LMS_SIC_Ind2"
            / f"LMS_SIC_1_digit_Ind2_from_{BASE_YEAR}_factors.hdf",
            df=df_wide,
            key=f"factors_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )


def pre_process_lms_soc_by_g(separate_by_g: bool) -> None:
    soc = []
    # Read in and format the LM&S data for each region
    for region in REGION_CORRESPONDENCE["RGN21NM"]:
        if separate_by_g:
            for g in ["Males", "Females"]:
                df = pd.read_csv(
                    LMS_INPUT_DIR / rf"LMS_SOC\LMS_Occ_T1_{g}_{region}.csv",
                    header=[0],
                    skiprows=[1, 11, 12],
                )
                df["g"] = g
        else:
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

    soc_rgns = pp.infill_for_years(
        df=soc_rgns, forecast_years=FORECAST_YEARS, extroplate_beyond_end="static"
    )

    # prepare for export
    soc_rgns = soc_rgns.astype({"SOC": "int"}).rename(columns={"SOC": "soc"})


    # Remap region back to codes
    soc_rgns["region"] = soc_rgns["region"].map(
        dict(
            (x, y)
            for y, x in dict(sorted(REGION_CORRESPONDENCE.values.tolist())).items()
        )
    )

    if separate_by_g:
        soc_rgns["g"] = soc_rgns["g"].map({"Males": 1, "Females": 2})
        soc_rgns = soc_rgns.groupby(["soc", "region", "g"]).sum().reset_index()
        soc_rgns[f"soc_prop_{BASE_YEAR}"] = soc_rgns[BASE_YEAR] / (
            soc_rgns.groupby(["region", "g"])[BASE_YEAR].transform("sum")
        )
    else:
        soc_rgns = soc_rgns.groupby(["soc", "region"]).sum().reset_index()
        soc_rgns[f"soc_prop_{BASE_YEAR}"] = soc_rgns[BASE_YEAR] / (
            soc_rgns.groupby(["region"])[BASE_YEAR].transform("sum")
        )

    # Output as hdf, ready to be read in as DVector
    for year in FORECAST_YEARS:

        if separate_by_g:
            soc_rgns[f"soc_prop_{year}"] = soc_rgns[year] / (
                soc_rgns.groupby(["region", "g"])[year].transform("sum")
            )
            index_columns = ["g", "soc"]
            out_stem = f"soc_pp_change_from_{BASE_YEAR}.hdf"
        else:
            soc_rgns[f"soc_prop_{year}"] = soc_rgns[year] / (
                soc_rgns.groupby(["region"])[year].transform("sum")
            )
            index_columns = ["soc"]
            out_stem = f"soc_no_g_pp_change_from_{BASE_YEAR}.hdf"

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
            index_cols=index_columns,
            value_column=f"soc_pp_change_from_{BASE_YEAR}_to_{year}",
        )
        pp.save_preprocessed_hdf(
            source_file_path=LMS_INPUT_DIR / "LMS_SOC" / out_stem,
            df=df_wide,
            key=f"change_from_{BASE_YEAR}_to_{year}",
            mode="a",
        )
        print(f"written for {year=}")


if __name__ == "__main__":
    main()
