from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

import land_use.preprocessing as pp


from land_use import logging as lu_logging

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants

# Parameters
BASE_YEAR = 2023
# Include the base year in here as we are pivoting from it
FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048, 2053]
# Location of base employment devector which we are pivoting from
BASE_EMP_DV = Path(
    r"F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs\Output E6.hdf"
)
LMS_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\Labour Market and Skills")
IPF_TARGET_OUT_DIR = Path(
    r"F:\Working\Land-Use\EMPLOYMENT_TARGETS\based_on_241213_Employment_test"
)

# This is fairly fixed and maps the GOR/RGN ONS codes to land use (NE, NW, ...., Lon, Scotland, Wales)
REGION_CORRESPONDENCE = pd.read_csv(
    Path(
        r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
        "GOR2021_CD_NM_EWS.csv",
    )
)

# Set up logging of key inputs as this helps audit trail
LOGGER = logging.getLogger(__name__)
LOGGER = lu_logging.configure_logger(
    output_dir=IPF_TARGET_OUT_DIR, log_name="reformat_employment_generation_log"
)

LOGGER.info(f"Process run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
LOGGER.info(f"{BASE_YEAR=}")
LOGGER.info(f"{FORECAST_YEARS=}")
LOGGER.info(f"{BASE_EMP_DV=}")
LOGGER.info(f"{LMS_INPUT_DIR=}")
LOGGER.info(f"{IPF_TARGET_OUT_DIR=}")

# REGION_CORRESPONDENCE doesn't need to be logged as it is fairly fixed over time.
IPF_TARGET_OUT_DIR.mkdir(exist_ok=True)


def return_base_as_rgn() -> DVector:
    base_dv = DVector.load(BASE_EMP_DV)

    base_dv_rgn = base_dv.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )

    return base_dv_rgn


def main():
    IPF_TARGET_OUT_DIR.mkdir(exist_ok=True)

    # TODO: consider if to move this to population as it is used there.
    # pre_process_lms_soc_by_g(separate_by_g=True)

    soc_pp_changes = pre_process_lms_soc_by_g(separate_by_g=False)

    base_dv_rgn = return_base_as_rgn()
    sic_factors_by_year = pre_process_lms_sic()

    for year in FORECAST_YEARS:

        # common hdf key across all outputs
        hdf_key = f"targets_{year}"

        # creating sic 1 digit targets, and finding totals
        sic_factors_for_year = sic_factors_by_year[f"{BASE_YEAR}_to_{year}"]

        sic_targets = calc_sic_targets(
            base_dv_rgn=base_dv_rgn,
            sic_factors=sic_factors_for_year,
            path_out=IPF_TARGET_OUT_DIR / "sic_targets.hdf",
            hdf_key=hdf_key,
        )

        yearly_target_total_except_soc_4 = sic_targets.sum()

        # now need to find the soc splits in the base
        soc_pp_change_for_year = soc_pp_changes[f"{BASE_YEAR}_to_{year}"]

        calc_soc_targets(
            base_dv=base_dv_rgn,
            target_total=yearly_target_total_except_soc_4,
            pp_change_from_base=soc_pp_change_for_year,
            path_out=IPF_TARGET_OUT_DIR / "soc_targets.hdf",
            hdf_key=f"targets_{year}",
        )


def calc_soc_targets(
    base_dv: DVector,
    target_total: float,
    pp_change_from_base: pd.DataFrame,
    path_out: None | Path = None,
    hdf_key: None | str = "df",
):

    base_soc_splits = return_base_soc_splits(base_dv)

    target_soc_splits = base_soc_splits + pp_change_from_base

    soc_targets = target_soc_splits * target_total

    if path_out:
        message = f"writing to {path_out}, with {hdf_key=}"
        LOGGER.info(message)
        soc_targets.to_hdf(path_out, key=hdf_key, mode="a")
    return soc_targets


def calc_sic_targets(
    base_dv_rgn: DVector,
    sic_factors: pd.DataFrame,
    path_out: Path | None = None,
    hdf_key: None | str = "df",
) -> DVector:
    """Calculate the sic targets and return. Optionally write to a hdf (if provided with a path_out)."""

    base_dv_rgn_sic_df = base_dv_rgn.aggregate(segs=["sic_1_digit"]).data
    after_sic_factors = base_dv_rgn_sic_df * sic_factors

    sic_targets = after_sic_factors.groupby(level="sic_1_digit").sum()
    # remove sic -1, which aligns with soc 4, and also remove 20,21 as they are empty
    if not path_out:
        return sic_targets

    sic_targets = sic_targets.loc[list(range(1, 20))]
    path_out = path_out
    message = f"writing to {path_out}, with {hdf_key=}"
    LOGGER.info(message)
    sic_targets.to_hdf(path_out, key=hdf_key, mode="a")

    return sic_targets


def return_base_soc_splits(base_dv_rgn):
    base_dv_rgn_soc_df = base_dv_rgn.aggregate(segs=["soc"]).data

    base_dv_rgn_soc_exc_4_df = base_dv_rgn_soc_df.loc[[1, 2, 3]]

    base_soc_splits = base_dv_rgn_soc_exc_4_df / base_dv_rgn_soc_exc_4_df.sum()
    return base_soc_splits


def pre_process_lms_sic() -> dict[str, pd.DataFrame]:
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

    # Output as dictionary, separated by years
    dfs_wide = {}
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
        df_wide = df_wide.fillna(1)

        dfs_wide[f"{BASE_YEAR}_to_{year}"] = df_wide
    return dfs_wide


def pre_process_lms_soc_by_g(separate_by_g: bool) -> dict[str, pd.DataFrame]:
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

    # Output as dictionary, separated by years
    dfs_wide = {}
    for year in FORECAST_YEARS:

        if separate_by_g:
            soc_rgns[f"soc_prop_{year}"] = soc_rgns[year] / (
                soc_rgns.groupby(["region", "g"])[year].transform("sum")
            )
            index_columns = ["g", "soc"]
        else:
            soc_rgns[f"soc_prop_{year}"] = soc_rgns[year] / (
                soc_rgns.groupby(["region"])[year].transform("sum")
            )
            index_columns = ["soc"]

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
        dfs_wide[f"{BASE_YEAR}_to_{year}"] = df_wide

    return dfs_wide


if __name__ == "__main__":
    main()
