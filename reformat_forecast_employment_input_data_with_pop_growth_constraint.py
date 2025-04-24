from pathlib import Path

import pandas as pd

from caf.base.data_structures import DVector

import land_use.preprocessing as pp

from caf.base.zoning import TranslationWeighting

from land_use import constants

REGION_CORRESPONDENCE = pd.read_csv(
    Path(
        r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
        "GOR2021_CD_NM_EWS.csv",
    )
)

BASE_YEAR = 2023
ONS_CROSSOVER_YEAR = 2043
LMS_INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import\Labour Market and Skills")
FORECAST_YEARS = [2023, 2028, 2033, 2038, 2043, 2048, 2053]
FORECAST_YEARS = [2023, 2033]

ONS_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\pop_projs")

IPF_TARGET_OUT_DIR = Path(
    r"F:\Working\Land-Use\EMPLOYMENT_TARGETS\based_on_241213_Employment_test_with_ons_pop_growth"
)
IPF_TARGET_OUT_DIR.mkdir(exist_ok=True)


def main():

    rgn_factors = calc_pop_rgn_factors()

    base_emp_rgn_dv = fetch_base_emp_in_rgn()

    base_pop_totals_rgn = calc_base_jobs_totals_rgn(base_emp_rgn=base_emp_rgn_dv)

    forecast_totals_rgn = rgn_factors.copy()

    for year in FORECAST_YEARS:
        factor_col = f"factor_{BASE_YEAR}_to_{year}"
        forecast_totals_rgn[f"jobs_target_{year}"] = (
            base_pop_totals_rgn[f"jobs_{BASE_YEAR}"] * rgn_factors[factor_col]
        )

    for year in FORECAST_YEARS:

        # common hdf key across all outputs
        hdf_key = f"targets_{year}"

        print(f"creating sic targets for {year}")
        calc_sic_targets(
            base_emp_rgn_dv=base_emp_rgn_dv,
            forecast_totals_rgn=forecast_totals_rgn,
            forecast_year=year,
            path_out=IPF_TARGET_OUT_DIR / "sic_targets.hdf",
            hdf_key=hdf_key,
        )

        print(f"creating soc targets for {year}")
        create_soc_targets(
            base_emp_rgn_dv=base_emp_rgn_dv,
            forecast_totals_rgn=forecast_totals_rgn,
            forecast_year=year,
            path_out=IPF_TARGET_OUT_DIR / "soc_targets.hdf",
            hdf_key=hdf_key,
        )


def calc_base_jobs_totals_rgn(base_emp_rgn: DVector) -> pd.DataFrame:
    df = base_emp_rgn.add_segments(["total"]).aggregate(["total"]).data
    df = df.transpose()
    df = df.reset_index(names="CODE")
    df = df.set_index("CODE")
    df = df.rename(columns={1: f"jobs_{BASE_YEAR}"})
    return df


def fetch_base_emp_in_rgn() -> DVector:
    base_emp_path = Path(
        r"F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs\Output E6.hdf"
    )
    base_emp = DVector.load(base_emp_path)

    base_emp_rgn = base_emp.translate_zoning(
        new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=False,
    )
    return base_emp_rgn


def calc_pop_rgn_factors() -> pd.DataFrame:
    non_english_rgn_pop_factors = calc_non_english_rgn_pop_factors()
    english_rgn_pop_factors = calc_english_rgn_pop_factors()

    rgn_factors = pd.concat([english_rgn_pop_factors, non_english_rgn_pop_factors])

    rgn_factors = rgn_factors.set_index(["CODE"])
    return rgn_factors


def calc_english_rgn_pop_factors() -> pd.DataFrame:
    gor_to_crossover = extract_england_region_pop_forecasts(
        filename=Path("2018_based_england_regions_pop_projections.xlsx")
    )

    england_factors_post_crossover = calc_england_factors_post_crossover()

    england_rgn_factors = calc_gor_factors_post_crossover(
        gor_to_crossover=gor_to_crossover,
        england_factors_post_crossover=england_factors_post_crossover,
    )

    england_rgn_factors = england_rgn_factors.set_index("CODE")

    keep_cols = [f"factor_{BASE_YEAR}_to_{year}" for year in FORECAST_YEARS]

    england_rgn_factors = england_rgn_factors[keep_cols]

    england_rgn_factors = england_rgn_factors.reset_index()
    return england_rgn_factors


def extract_england_region_pop_forecasts(filename: Path) -> pd.DataFrame:
    """Extract persons projects for english regions. Not separated by gender."""
    df = pd.read_excel(ONS_DIR / filename, sheet_name="Persons", skiprows=6)
    df = df[~df["AGE GROUP"].isin(["All ages", "90+"])]

    df[["from_age", "to_age"]] = df["AGE GROUP"].str.split("-", expand=True)

    df = df.drop(columns=["AGE GROUP", "AREA"])

    df["from_age"] = df["from_age"].astype(int)
    df["to_age"] = df["to_age"].astype(int)

    df_15_19 = df[df["from_age"] == 15]
    df_20_74 = df[(df["from_age"] >= 20) & (df["to_age"] <= 74)]

    df_15_19_totals = df_15_19.groupby("CODE").sum()
    df_20_74_totals = df_20_74.groupby("CODE").sum()

    # want to remove a fifth of the 15-19 age range
    df_working_age = df_15_19_totals * 0.8 + df_20_74_totals

    df_working_age = df_working_age.drop(columns=["from_age", "to_age"])

    # need to be a little bit smarter for which years to calculate factors for,
    # we definitely need up to the crossover
    factor_years = FORECAST_YEARS.copy()
    if not ONS_CROSSOVER_YEAR in factor_years:
        factor_years.append(ONS_CROSSOVER_YEAR)

    # and then we should cap it at the crossover
    factor_years = [year for year in factor_years if year <= ONS_CROSSOVER_YEAR]

    # get factors from base year
    for col in factor_years:
        df_working_age[f"factor_{BASE_YEAR}_to_{col}"] = (
            df_working_age[col] / df_working_age[BASE_YEAR]
        )
    df_working_age = df_working_age.reset_index()

    # remove england as want regions
    df_working_age = df_working_age[df_working_age["CODE"] != "E92000001"]

    return df_working_age


def calc_england_factors_post_crossover() -> pd.DataFrame:

    df = extract_country_level_pop_forecasts(
        "2021_based_interim_england_pop_projections.xlsx", country_code="E92000001"
    )
    df = df.drop(columns=["CODE"])
    keep_cols = [col for col in df.columns if col >= ONS_CROSSOVER_YEAR]
    df = df[keep_cols]
    for year in df:
        df[f"factor_{ONS_CROSSOVER_YEAR}_to_{year}"] = df[year] / df[ONS_CROSSOVER_YEAR]
    return df


def calc_gor_factors_post_crossover(
    gor_to_crossover: pd.DataFrame, england_factors_post_crossover: pd.DataFrame
) -> pd.DataFrame:
    merged = pd.merge(gor_to_crossover, england_factors_post_crossover, how="cross")
    # multiply the factor_2043_to_xxxxx by factor_2023_to_2043 to give factor_2023_to_2043
    post_crossover_cols = [
        col for col in merged if str(col).startswith(f"factor_{ONS_CROSSOVER_YEAR}")
    ]
    for col in post_crossover_cols:
        year = str(col).split("_")[-1]
        uplift_to_crossover = f"factor_{BASE_YEAR}_to_{ONS_CROSSOVER_YEAR}"
        forecast_year_factor = f"factor_{BASE_YEAR}_to_{year}"
        merged[forecast_year_factor] = merged[col] * merged[uplift_to_crossover]
    return merged


def calc_non_english_rgn_pop_factors() -> pd.DataFrame:
    scotland_pop_forecast = extract_country_level_pop_forecasts(
        "2020_based_interim_scotland_pop_projections.xlsx", country_code="S92000003"
    )
    wales_pop_forecast = extract_country_level_pop_forecasts(
        "2021_based_interim_wales_pop_projections.xlsx", country_code="W92000004"
    )

    combined_pop = pd.concat([scotland_pop_forecast, wales_pop_forecast])

    for col in FORECAST_YEARS:
        combined_pop[f"factor_{BASE_YEAR}_to_{col}"] = (
            combined_pop[col] / combined_pop[BASE_YEAR]
        )

    keep_cols = [f"factor_{BASE_YEAR}_to_{col}" for col in FORECAST_YEARS]

    keep_cols.append("CODE")

    combined_pop = combined_pop[keep_cols]

    return combined_pop


def extract_country_level_pop_forecasts(
    filename: str, country_code: str
) -> pd.DataFrame:
    # calculate population growth from ons
    df = pd.read_excel(ONS_DIR / filename, sheet_name="Population")

    df = df[~df["Age"].isin(["105 - 109", "110 and over"])]

    df["Age"] = df["Age"].astype(int)

    df_working_age = df[(df["Age"] < 75) & (df["Age"] > 15)]

    totals = df_working_age.sum(axis=0).reset_index(name="total")
    totals = totals.rename(columns={"index": "age"})

    totals = totals[~totals["age"].isin(["Sex", "Age"])]
    totals = totals.rename(columns={"age": "year", "total": country_code})
    totals["year"] = totals["year"].astype(int)
    totals = totals.set_index("year")
    totals = totals.transpose()

    totals = totals.reset_index(names="CODE")

    return totals


def calc_sic_targets(
    base_emp_rgn_dv: DVector,
    forecast_totals_rgn: pd.DataFrame,
    forecast_year: int,
    path_out: Path,
    hdf_key: None | str = "df",
) -> DVector:

    base_sic_values = base_emp_rgn_dv.aggregate(["sic_1_digit"]).data
    base_sic_props = base_sic_values.div(base_sic_values.sum(axis=0), axis=1)

    sic_rgns = pre_process_lms_sic()

    forecast_sic_changes = sic_rgns.reset_index()

    for col in forecast_sic_changes.columns:
        if isinstance(col, int):
            forecast_sic_changes[col] = forecast_sic_changes.groupby("region")[
                col
            ].transform(lambda x: x / x.sum())

    for col in forecast_sic_changes.columns:
        if isinstance(col, int):
            forecast_sic_changes[f"{BASE_YEAR}_to_{col}_pp_change"] = (
                forecast_sic_changes[col] - forecast_sic_changes[BASE_YEAR]
            )

    sic_changes_wide = pd.pivot(
        forecast_sic_changes.reset_index(),
        index="sic_1_digit",
        columns="region",
        values=f"{BASE_YEAR}_to_{forecast_year}_pp_change",
    )

    forecast_sic_prop = base_sic_props + sic_changes_wide

    # infill where the nas are, which will be the missing sic_1_digits from the forecast
    forecast_sic_prop = forecast_sic_prop.fillna(base_sic_props)

    forecast_sic_prop_t = forecast_sic_prop.transpose()

    sic_targets = forecast_sic_prop_t.copy()

    for col in sic_targets:
        sic_targets[col] = (
            sic_targets[col] * forecast_totals_rgn[f"jobs_target_{forecast_year}"]
        )

    sic_targets_dv_format = sic_targets.transpose()

    # TODO consider if this can be/should be dynamic?
    # remove sic_1_digit 20 and 21 as they are empty
    # TODO: testing if removing -1 allows IPF to run
    sic_targets_dv_format = sic_targets_dv_format.drop(index=[-1, 20, 21])

    message = f"writing to {path_out}, with {hdf_key=}"
    print(message)
    # LOGGER.info(message)
    sic_targets_dv_format.to_hdf(path_out, key=hdf_key, mode="a")

    return sic_targets_dv_format


def pre_process_lms_sic():
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

    sic_rgns = (
        sic_rgns[["LU_SIC_1_digit", "region"] + years]
        .groupby(by=["LU_SIC_1_digit", "region"], as_index=False)[years]
        .sum()
    )
    # TODO numbers under 10,000?

    forecast_years = FORECAST_YEARS
    if not BASE_YEAR in FORECAST_YEARS:
        forecast_years.append(BASE_YEAR)

    sic_rgns = pp.infill_for_years(
        df=sic_rgns, forecast_years=forecast_years, extroplate_beyond_end="static"
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

    df_long = sic_rgns.melt(
        id_vars=["sic_1_digit", "region"], var_name="year", value_name="jobs"
    )

    df_wide = df_long.pivot(
        index=["sic_1_digit", "region"], columns=["year"], values="jobs"
    )

    return df_wide


def create_soc_targets(
    base_emp_rgn_dv: DVector,
    forecast_totals_rgn: pd.DataFrame,
    forecast_year: int,
    path_out: Path,
    hdf_key: None | str = "df",
) -> pd.DataFrame:

    df = pre_process_lms_soc()

    soc_changes = pd.pivot(
        data=df,
        index=["soc"],
        columns=["region"],
        values=f"soc_pp_change_from_{BASE_YEAR}_to_{forecast_year}",
    )

    # # work out soc base splits
    base_soc_values = base_emp_rgn_dv.aggregate(["soc"]).data
    base_soc_props = base_soc_values.div(base_soc_values.sum(axis=0), axis=1)

    forecast_soc_props = base_soc_props + soc_changes

    forecast_soc_props = forecast_soc_props.fillna(base_soc_props)

    forecast_soc_props_t = forecast_soc_props.transpose()

    soc_targets = forecast_soc_props_t.copy()

    for col in soc_targets:
        soc_targets[col] = (
            soc_targets[col] * forecast_totals_rgn[f"jobs_target_{forecast_year}"]
        )

    soc_targets_dv_format = soc_targets.transpose()

    # TODO: testing if removing soc4 allows IPF to run
    soc_targets_dv_format = soc_targets_dv_format.drop(index=[4])

    message = f"writing to {path_out}, with {hdf_key=}"
    print(message)
    # LOGGER.info(message)
    soc_targets_dv_format.to_hdf(path_out, key=hdf_key, mode="a")

    return soc_targets_dv_format


def pre_process_lms_soc() -> pd.DataFrame:
    soc = []
    # Read in and format the LM&S data for each region
    for region in REGION_CORRESPONDENCE["RGN21NM"]:
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

    forecast_years_inc_base = FORECAST_YEARS
    if not BASE_YEAR in FORECAST_YEARS:
        forecast_years_inc_base.append(BASE_YEAR)

    soc_rgns = pp.infill_for_years(
        df=soc_rgns,
        forecast_years=forecast_years_inc_base,
        extroplate_beyond_end="static",
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

    soc_rgns = soc_rgns.groupby(["soc", "region"]).sum().reset_index()
    soc_rgns[f"soc_prop_{BASE_YEAR}"] = soc_rgns[BASE_YEAR] / (
        soc_rgns.groupby(["region"])[BASE_YEAR].transform("sum")
    )

    # Output as dictionary, separated by years
    for year in FORECAST_YEARS:

        soc_rgns[f"soc_prop_{year}"] = soc_rgns[year] / (
            soc_rgns.groupby(["region"])[year].transform("sum")
        )

        soc_rgns[f"soc_pp_change_from_{BASE_YEAR}_to_{year}"] = (
            soc_rgns[f"soc_prop_{year}"] - soc_rgns[f"soc_prop_{BASE_YEAR}"]
        )

    return soc_rgns


if __name__ == "__main__":
    main()
