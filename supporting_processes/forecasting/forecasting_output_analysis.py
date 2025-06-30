from pathlib import Path

import pandas as pd
import numpy as np

from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting

from land_use import constants, data_processing
import land_use.preprocessing as pp

# Output directories
POP_OUTPUT_DIR = Path(r"F:\Working\Land-Use\forecast_population_20250626")
EMP_OUTPUT_DIR = Path(r"F:\Working\Land-Use\temp_forecast_employment_testing_moving_to_config_with_ons_pop_growth_all_sic")
POP_ANALYSIS_DIR = Path(
    r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\pop"
)
EMP_ANALYSIS_DIR = Path(
    r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\emp"
)

HOUSEHOLDS_ONS_DIR = Path(r"I:\NorMITs Land Use\2023\import\ONS\forecasting\hh_projs")

rgn_code_to_desc = {
    "E12000001": "NE",
    "E12000002": "NW",
    "E12000003": "YH",
    "E12000004": "EM",
    "E12000005": "WM",
    "E12000006": "EoE",
    "E12000007": "Lon",
    "E12000008": "SE",
    "E12000009": "SW",
    "S92000003": "Scotland",
    "W92000004": "Wales",
}

region_mapping = {
        "W92000004": "Wales",
        "E12000008": "South East",
        "E12000004": "East Midlands",
        "E12000005": "West Midlands",
        "E12000002": "North West",
        "E12000009": "South West",
        "E12000007": "London",
        "E12000003": "Yorkshire and The Humber",
        "E12000001": "North East",
        "E12000006": "East of England",
        "S92000003": "Scotland",
    }


def summarise_population_outputs(output_file_name: str, regions: list, years_to_extract: list):
    """
    Function to summarise the hdf outputs from the forecast population process in "forecast_population.py"

    Parameters
    ----------
    output_file_name: string
        Name of the csv file to output to
    regions: list
        Regions to analyse for
    years_to_extract: list
        Forecast years to analyse
    """
    final_dfs = []
    for year in years_to_extract:
        for rgn in regions:
            print(f"Summarising for {year}, {rgn}")
            dv = DVector.load(
                Path(
                    POP_OUTPUT_DIR / rf"02_Final Outputs\Output Pop_"
                    rf"{rgn}_{year}.hdf"
                )
            )
            dv_translated = dv.translate_zoning(
                new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.SPATIAL,
                check_totals=False,
            )
            for seg in dv_translated.segmentation.names:
                print(seg)
                dv_seg = dv_translated.aggregate([seg])

                df = dv_seg.data.stack().reset_index()
                df["seg"] = seg
                df = df.rename(columns={seg: "seg_value"}).set_axis(["seg_value", "region", "value", "seg"], axis=1)

                df["year"] = year
                final_dfs.append(df)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    final_output["region"] = final_output["region"].map(region_mapping)
    final_output.to_csv(POP_ANALYSIS_DIR / f"{output_file_name}.csv")


def summarise_pop_and_hh_outputs_multi_segments(
        output_file_name: str,
        segments_to_summarise: list,
        years_to_summarise: list,
        regions: list,
        output_file_prefixes: list,
        forecast_pop_output_dir: Path,
        output_dir: Path
):
    """
    Function to summarise the hdf outputs from the forecast population process across multiple segments,
    useful for cross-table analysis

    Parameters
    ----------
    output_file_name: string
        Name of the csv file to output to
    segments_to_summarise: list
        Segments in the output data to summarise
    years_to_summarise: list
        Forecast years to analyse
    regions: list
        Regions to analyse
    output_file_prefixes: list
        Prefix of the population output files e.g. pop, households
    forecast_pop_output_dir: path
        Folder of forecast population outputs
    output_dir: path
        Folder where to save the csv summaries
    """
    final_dfs = []
    for file in output_file_prefixes:
        print(f"Summarising for {file}")
        for year in years_to_summarise:
            print(f"Summarising for {year}")
            for rgn in regions:
                print(f"Summarising for {year}, {rgn}")
                if year == 2023:
                    if file == "Pop":
                        base_pop_age_ntem_path = Path(r"F:\Working\Land-Use\BASE_POPULATION_WITH_AGE_NTEM")
                        dv = DVector.load(
                            Path(base_pop_age_ntem_path / fr"based_on_241220_Populationv2\Output P11_{rgn}.hdf"))
                    else:
                        dv = DVector.load(
                            Path(
                                fr"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs\Output P13.3_{rgn}.hdf"
                                )
                        )
                else:
                    dv = DVector.load(
                        Path(
                            forecast_pop_output_dir / rf"02_Final Outputs\Output {file}_{rgn}_{year}.hdf"
                        )
                    )
                dv_translated = dv.translate_zoning(
                    new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                    cache_path=constants.CACHE_FOLDER,
                    weighting=TranslationWeighting.SPATIAL,
                    check_totals=False,
                )

                print(f"Aggregating for {segments_to_summarise}")
                dv_seg = dv_translated.aggregate(segments_to_summarise)

                df = dv_seg.data.stack().reset_index()
                df = df.set_axis(segments_to_summarise + ["region", "value"], axis=1)
                df = df[df["value"] != 0]

                df["output"] = file
                df["year"] = year
                final_dfs.append(df)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    final_output["region"] = final_output["region"].map(region_mapping)
    final_output.to_csv(output_dir / f"{output_file_name}.csv", index=False)


def summarise_emp_outputs(output_file_name: str, years_to_extract: list):
    """
    Function to summarise the hdf outputs from the forecast employment process in "forecast_employment.py"

    Parameters
    ----------
    output_file_name: string
        Name of the csv file to output to
    years_to_extract: list
        Forecast years to analyse
    """
    final_dfs = []
    for year in years_to_extract:
        print(f"Summarising for {year}")
        dv = DVector.load(
            Path(
                EMP_OUTPUT_DIR / rf"02_Final Outputs\Output Emp_"
                rf"{year}.hdf"
            )
        )

        dv_translated = dv.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False,
        )

        # aggregate segmentation
        dv1 = dv_translated.aggregate(["sic_1_digit"])
        dv2 = dv_translated.aggregate(["sic_2_digit"])
        dv3 = dv_translated.aggregate(["sic_4_digit"])
        dv4 = dv_translated.aggregate(["soc"])

        # stack columns and format
        dv1 = (
            dv1.data.stack()
            .reset_index()
            .set_axis(["segment", "region", "value"], axis=1)
        )
        dv1["segmentation"] = "sic_1_digit"
        dv1["year"] = year
        dv1 = dv1[["year", "segmentation", "region", "segment", "value"]]

        # stack columns and format
        dv2 = (
            dv2.data.stack()
            .reset_index()
            .set_axis(["segment", "region", "value"], axis=1)
        )
        dv2["segmentation"] = "sic_2_digit"
        dv2["year"] = year
        dv2 = dv2[["year", "segmentation", "region", "segment", "value"]]

        # stack columns and format
        dv3 = (
            dv3.data.stack()
            .reset_index()
            .set_axis(["segment", "region", "value"], axis=1)
        )
        dv3["segmentation"] = "sic_4_digit"
        dv3["year"] = year
        dv3 = dv3[["year", "segmentation", "region", "segment", "value"]]

        # stack columns and format
        dv4 = (
            dv4.data.stack()
            .reset_index()
            .set_axis(["segment", "region", "value"], axis=1)
        )
        dv4["segmentation"] = "soc"
        dv4["year"] = year
        dv4 = dv4[["year", "segmentation", "region", "segment", "value"]]

        final_dfs.append(dv1)
        final_dfs.append(dv2)
        final_dfs.append(dv3)
        final_dfs.append(dv4)

    final_output = pd.concat(final_dfs)
    # redefine the region names
    final_output["region"] = final_output["region"].map(region_mapping)
    final_output.to_csv(EMP_ANALYSIS_DIR / f"{output_file_name}.csv")


def summarise_household_outputs(output_file_name: str, regions: list, years_to_extract: list):
    """
    Function to summarise the hdf outputs from the forecast households process in "forecast_population.py"

    Parameters
    ----------
    output_file_name: string
        Name of the csv file to output to
    regions: list
        Regions to analyse for
    years_to_extract: list
        Forecast years to analyse
    """
    final_dfs = []
    for year in years_to_extract:
        for rgn in regions:
            print(f"Summarising for {year}, {rgn}")
            if year == 2023:
                dv = DVector.load(
                    Path(fr"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs\Output P13.3_{rgn}.hdf"))
            else:
                dv = DVector.load(POP_OUTPUT_DIR / fr"02_Final Outputs\Output Households_{rgn}_{year}.hdf")

            dv_translated = dv.translate_zoning(
                new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.SPATIAL,
                check_totals=False
            )

            for seg in dv_translated.segmentation.names:
                print(seg)
                # aggregate segmentation
                df = dv_translated.aggregate([seg])
                df = df.data.reset_index()

                df["seg"] = seg
                df = df.rename(columns={seg: "seg_value"})

                df = df.rename(columns=rgn_code_to_desc)
                df_long = df.melt(id_vars=["seg", "seg_value"], var_name="rgn")

                df_long["year"] = year

                final_dfs.append(df_long)

    final_output = pd.concat(final_dfs)

    final_output.to_csv(POP_ANALYSIS_DIR / f"{output_file_name}.csv")


def dvector_segment_comparisons(
        dvector_dict: dict,
        output_file_name: Path,
        forecast_type: str
):
    """
    Function to compare segment proportions between multiple DVectors

    Parameters
    ----------
    dvector_dict: dictionary
        DVectors for comparison, includes full path to the DVector
    output_file_name: Path
        Name of the csv file to output to
    forecast_type: string
    """
    # %%
    single_seg_totals_and_prop = []
    for dv_name, dv_path in dvector_dict.items():
        dv = DVector.load(Path(dv_path))

        dv_rgn = dv.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False,
        )
        # %%
        for seg in dv_rgn.segmentation.names:
            print(seg)
            dv_seg = dv_rgn.aggregate([seg])

            df = dv_seg.data.reset_index()

            df["seg"] = seg
            df = df.rename(columns={seg: "seg_value"})

            df = df.rename(columns=rgn_code_to_desc)
            df_long = df.melt(id_vars=["seg", "seg_value"], var_name="rgn")

            df_long["prop"] = df_long["value"] / df_long.groupby("rgn")[
                "value"
            ].transform("sum")

            if forecast_type == "emp":
                if seg == "soc":
                    df_long_exc_some = df_long[df_long["seg_value"] != 4]
                if seg in ["sic_1_digit", "sic_2_digit", "sic_4_digit"]:
                    df_long_exc_some = df_long[df_long["seg_value"] > 0]

                df_long_exc_some = df_long_exc_some.copy()

                df_long_exc_some["prop_excluding"] = df_long_exc_some[
                    "value"
                ] / df_long_exc_some.groupby("rgn")["value"].transform("sum")

                df_combined = pd.merge(df_long, df_long_exc_some, how="outer")
            else:
                df_combined = df_long.copy()
                df_combined["output_name"] = dv_name

            single_seg_totals_and_prop.append(df_combined)

    single_seg_df = pd.concat(single_seg_totals_and_prop)

    print(single_seg_df)
    single_seg_df.to_csv(output_file_name)


def calculate_occupancies(
        base_pop_path: Path,
        forecast_pop_path: Path,
        regions: list,
        forecast_years: list,
        agg_segments: list,
        output_file_name: str
):
    """
    Function to calculate occupancies (population / households) for a defined segment
    Outputs at LSOA level and at GOR level

    Parameters
    ----------
    base_pop_path: Path
        Base population DVector output path
    forecast_pop_path: Path
        Forecast population DVector output path
    regions: list
        Regions to analyse for
    forecast_years: list
        Forecast years to analyse
    agg_segments:
        Segments to aggregate the data and outputs to (can be 1 segment or 2 segments)
    output_file_name:
        Name of the csv file to output to
    """

    base = []
    base_rgn = []
    f_years = []
    f_years_rgn = []
    for rgn in regions:
        print(fr"Calculating for {rgn}")
        base_pop = DVector.load(base_pop_path / fr"Output P11_{rgn}.hdf")
        base_hh = DVector.load(base_pop_path / fr"Output P13.3_{rgn}.hdf")

        # Add total segment to base population
        if "total" in agg_segments:
            if "total" not in base_pop.segmentation.names:
                base_pop = base_pop.add_segments(["total"])

            if "total" not in base_hh.segmentation.names:
                base_hh = base_hh.add_segments(["total"])

        base_pop_agg = base_pop.aggregate(agg_segments)
        base_hh_agg = base_hh.aggregate(agg_segments)

        base_pop_agg_rgn = base_pop_agg.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False
        )
        base_hh_agg_rgn = base_hh_agg.translate_zoning(
            new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
            cache_path=constants.CACHE_FOLDER,
            weighting=TranslationWeighting.SPATIAL,
            check_totals=False
        )

        base_occs = base_pop_agg / base_hh_agg
        base_occs = base_occs.data.T.reset_index(names="LSOA2021")
        if len(agg_segments) == 2:
            base_occs.columns = ["{} / {}".format(x, y) for x, y in base_occs.columns]
        base_occs['year'] = 2023

        base_occs_rgn = base_pop_agg_rgn / base_hh_agg_rgn
        base_occs_rgn = base_occs_rgn.data.T.reset_index(names="region")
        if len(agg_segments) == 2:
            base_occs_rgn.columns = ["{} / {}".format(x, y) for x, y in base_occs_rgn.columns]
        base_occs_rgn['year'] = 2023

        base.append(base_occs)
        base_rgn.append(base_occs_rgn)

        for year in forecast_years:
            print(fr"Calculating for {year}")
            forecast_pop = DVector.load(forecast_pop_path / fr"Output Pop_{rgn}_{year}.hdf")
            forecast_hh = DVector.load(forecast_pop_path / fr"Output Households_{rgn}_{year}.hdf")
            # Add total segment to base population
            if "total" in agg_segments:
                if "total" not in forecast_pop.segmentation.names:
                    forecast_pop = forecast_pop.add_segments(["total"])
                if "total" not in forecast_hh.segmentation.names:
                    forecast_hh = forecast_hh.add_segments(["total"])

            forecast_pop_agg = forecast_pop.aggregate(agg_segments)
            forecast_hh_agg = forecast_hh.aggregate(agg_segments)

            forecast_pop_agg_rgn = forecast_pop_agg.translate_zoning(
                new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.SPATIAL,
                check_totals=False
            )
            forecast_hh_agg_rgn = forecast_hh_agg.translate_zoning(
                new_zoning=constants.RGN_EWS_ZONING_SYSTEM,
                cache_path=constants.CACHE_FOLDER,
                weighting=TranslationWeighting.SPATIAL,
                check_totals=False
            )

            forecast_occs = forecast_pop_agg / forecast_hh_agg
            forecast_occs = forecast_occs.data.T.reset_index(names="LSOA2021")
            if len(agg_segments) == 2:
                forecast_occs.columns = ["{} / {}".format(x, y) for x, y in forecast_occs.columns]
            forecast_occs['year'] = year

            forecast_occs_rgn = forecast_pop_agg_rgn / forecast_hh_agg_rgn
            forecast_occs_rgn = forecast_occs_rgn.data.T.reset_index(names="region")
            if len(agg_segments) == 2:
                forecast_occs_rgn.columns = ["{} / {}".format(x, y) for x, y in forecast_occs_rgn.columns]
            forecast_occs_rgn['year'] = year

            f_years.append(forecast_occs)
            f_years_rgn.append(forecast_occs_rgn)

    # Format for output
    output_base = pd.concat(base)
    output_forecast = pd.concat(f_years)
    output = pd.concat([output_base, output_forecast])
    output.to_csv(
        Path(fr'F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\{output_file_name}.csv'),
        index=False,
        header=True
    )

    output_base_rgn = pd.concat(base_rgn)
    output_forecast_rgn = pd.concat(f_years_rgn)
    output_rgn = pd.concat([output_base_rgn, output_forecast_rgn])
    output_rgn.to_csv(
        Path(fr'F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\{output_file_name}_rgn.csv'),
        index=False,
        header=True
    )


def summarise_hh_nssec_targets():
    """
    Function to summarise the household NS-SEC targets generated from the forecast population outputs
    """
    final_dfs = []
    for year in ["2033", "2038", "2043", "2048", "2053"]:
        print(year)
        dv = data_processing.read_dvector_data(
            file_path=Path(
                r"F:\Working\Land-Use\POPULATION_TARGETS\based_on_241220_Populationv2\hh_ns-sec_targets.hdf"),
            geographical_level="LSOA2021",
            input_segments=["ns_sec"],
            hdf_key=f"targets_{year}"
        )
        dv_translated = dv.translate_zoning(new_zoning=constants.KNOWN_GEOGRAPHIES.get("RGN2021"),
                                            cache_path=constants.CACHE_FOLDER,
                                            weighting=TranslationWeighting.NO_WEIGHT)
        for seg in dv_translated.segmentation.names:
            print(seg)
            # aggregate segmentation
            df = dv_translated.aggregate([seg])
            df = df.data.reset_index()

            df["seg"] = seg
            df = df.rename(columns={seg: "seg_value"})

            df = df.rename(columns=region_mapping)
            df_long = df.melt(id_vars=["seg", "seg_value"], var_name="rgn")

            df_long["year"] = year

            final_dfs.append(df_long)

    final_output = pd.concat(final_dfs)
    final_output.to_csv(POP_ANALYSIS_DIR / f"hh_ns-sec_targets.csv")


def fetch_region_correspondence() -> pd.DataFrame:

    return pd.read_csv(
        Path(
            r"I:\NorMITs Land Use\2023\import\ONS\Correspondence_lists",
            "GOR2021_CD_NM_EWS.csv",
        )
    )


def summarise_ons_hh_projections():
    rgn_corr = fetch_region_correspondence()

    # ENGLAND
    hh_eng = pd.read_excel(
        HOUSEHOLDS_ONS_DIR / "england_LAs_2018_based_hh_projections.xlsx",
        sheet_name="406",
        header=4,
    ).rename(columns={"Area code": "region"})
    hh_eng = hh_eng.iloc[:, np.r_[0:2, 19:45]]
    # Filter to the defined regions (2018-based ONS data is available up to 2043)
    hh_eng = hh_eng[hh_eng["Area name"].isin(rgn_corr["RGN21NM"].tolist())].drop(
        columns=["Area name"]
    )

    # SCOTLAND
    hh_scot = pd.read_excel(
        HOUSEHOLDS_ONS_DIR / "scotland_2018-based_hh_projections.xlsx",
        sheet_name="Table 1",
        header=3,
        nrows=1,
    )
    hh_scot = hh_scot.iloc[:, 1:27]
    hh_scot.columns = hh_scot.columns.astype(int)
    # Define region code
    hh_scot["region"] = "S92000003"

    # WALES
    hh_wales = pd.read_csv(
        HOUSEHOLDS_ONS_DIR / "wales_2018_based_hh_projections.csv", header=5, nrows=1
    )
    hh_wales = hh_wales.drop(hh_wales.columns[0:2], axis=1)
    hh_wales.columns = hh_wales.columns.str.strip().astype(int)
    # Define region code
    hh_wales["region"] = "W92000004"

    # Join together England regions, Scotland and Wales 2018-based household data
    hh_projs = pd.concat([hh_eng, hh_scot, hh_wales])

    hh_projs = pp.infill_for_years(
        df=hh_projs, forecast_years=[2023, 2033, 2038, 2043, 2048, 2053], extroplate_beyond_end="trend"
    )

    hh_projs["region_name"] = hh_projs["region"].map(region_mapping)
    hh_projs = hh_projs.copy().drop(columns=["region"])

    hh_north = hh_projs[hh_projs["region_name"].isin(["North East", "North West", "Yorkshire and The Humber"])]
    hh_north = hh_north[[2023, 2033, 2038, 2043, 2048, 2053]].sum().to_frame().T
    hh_north["region_name"] = "The North"

    hh_eng = hh_projs[hh_projs["region_name"].isin(["North East", "North West", "Yorkshire and The Humber",
                                                    "East Midlands", "West Midlands", "East of England",
                                                    "London", "South East", "South West"])]
    hh_eng = hh_eng[[2023, 2033, 2038, 2043, 2048, 2053]].sum().to_frame().T
    hh_eng["region_name"] = "England"

    hh_all = hh_projs[[2023, 2033, 2038, 2043, 2048, 2053]].sum().to_frame().T
    hh_all["region_name"] = "GB"

    hh_projs = pd.concat([hh_projs, hh_north, hh_eng, hh_all])
    hh_projs = hh_projs[["region_name", 2023, 2033, 2038, 2043, 2048, 2053]]

    hh_projs.to_csv(POP_ANALYSIS_DIR / "ONS_household_projections_summary.csv", index=False)


def combine_outputs_to_summary_regions(
        pop_summary_csv_path: Path,
        emp_summary_csv_path: Path,
        hh_summary_csv_path: Path,
        pop_output_csv_path: Path,
        emp_output_csv_path: Path,
        hh_output_csv_path: Path
):
    """
    Function to combine the population, employment and household outputs (already processed to csvs) into
    aggregate regions (e.g. The North, England, GB)

    pop_summary_csv_path: Path
        File location of the already summarised population summary output
    emp_summary_csv_path: Path
        File location of the already summarised employment summary output
    hh_summary_csv_path: Path
        File location of the already summarised household summary output
    pop_output_csv_path: Path
        File location to save the population output summary to
    emp_output_csv_path: Path
        File location to save the employment output summary to
    hh_output_csv_path: Path
        File location to save the household output summary to
    """
    pop = pd.read_csv(pop_summary_csv_path)
    emp = pd.read_csv(emp_summary_csv_path)
    hh = pd.read_csv(hh_summary_csv_path)

    # Population
    pop_north = pop[pop["region"].isin(["North East", "North West", "Yorkshire and The Humber"])]
    pop_north = pop_north.groupby(["seg_value", "seg", "year"], as_index=False)[["value"]].sum()
    pop_north["region"] = "The North"
    pop_north = pop_north[["seg_value", "region", "value", "seg", "year"]]

    pop_eng = pop[pop["region"].isin(["North East", "North West", "Yorkshire and The Humber",
                                      "East Midlands", "West Midlands", "East of England",
                                      "London", "South East", "South West"])]
    pop_eng = pop_eng.groupby(["seg_value", "seg", "year"], as_index=False)[["value"]].sum()
    pop_eng["region"] = "England"
    pop_eng = pop_eng[["seg_value", "region", "value", "seg", "year"]]

    pop_all = pop.groupby(["seg_value", "seg", "year"], as_index=False)[["value"]].sum()
    pop_all["region"] = "GB"
    pop_all = pop_all[["seg_value", "region", "value", "seg", "year"]]

    # Employment
    emp_north = emp[emp["region"].isin(["North East", "North West", "Yorkshire and The Humber"])]
    emp_north = emp_north.groupby(["segmentation", "segment", "year"], as_index=False)[["value"]].sum()
    emp_north["region"] = "The North"
    emp_north = emp_north[["year", "segmentation", "region", "segment", "value"]]

    emp_eng = emp[emp["region"].isin(["North East", "North West", "Yorkshire and The Humber",
                                      "East Midlands", "West Midlands", "East of England",
                                      "London", "South East", "South West"])]
    emp_eng = emp_eng.groupby(["segmentation", "segment", "year"], as_index=False)[["value"]].sum()
    emp_eng["region"] = "England"
    emp_eng = emp_eng[["year", "segmentation", "region", "segment", "value"]]

    emp_all = emp.groupby(["segmentation", "segment", "year"], as_index=False)[["value"]].sum()
    emp_all["region"] = "GB"
    emp_all = emp_all[["year", "segmentation", "region", "segment", "value"]]

    # Households
    hh_north = hh[hh["rgn"].isin(["NE", "NW", "YH"])]
    hh_north = hh_north.groupby(["seg_value", "seg", "year"], as_index=False)[["value"]].sum()
    hh_north["rgn"] = "The North"
    hh_north = hh_north[["seg", "seg_value", "rgn", "value", "year"]]

    hh_eng = hh[hh["rgn"].isin(["NE", "NW", "YH",
                                "EM", "WM", "EoE",
                                "Lon", "SE", "SW"])]
    hh_eng = hh_eng.groupby(["seg_value", "seg", "year"], as_index=False)[["value"]].sum()
    hh_eng["rgn"] = "England"
    hh_eng = hh_eng[["seg", "seg_value", "rgn", "value", "year"]]

    hh_all = hh.groupby(["seg_value", "seg", "year"], as_index=False)[["value"]].sum()
    hh_all["rgn"] = "GB"
    hh_all = hh_all[["seg", "seg_value", "rgn", "value", "year"]]

    pop_output = pd.concat([pop_north, pop_eng, pop_all])
    emp_output = pd.concat([emp_north, emp_eng, emp_all])
    hh_output = pd.concat([hh_north, hh_eng, hh_all])

    pop_output.to_csv(pop_output_csv_path)
    emp_output.to_csv(emp_output_csv_path)
    hh_output.to_csv(hh_output_csv_path)


def combine_base_to_summary_regions(
        base_pop_summary_csv_path: Path,
        base_emp_summary_csv_path: Path,
        pop_output_csv_path: Path,
        emp_output_csv_path: Path
):
    """
    Function to combine the population, employment and household outputs (already processed to csvs) into
    aggregate regions (e.g. The North, England, GB)

    base_pop_summary_csv_path: Path
        File location of the already summarised population summary output
    base_emp_summary_csv_path: Path
        File location of the already summarised employment summary output
    output_csv_path: Path
        File location to save the output summary to
    """
    # Base
    pop = pd.read_csv(base_pop_summary_csv_path)
    emp = pd.read_csv(base_emp_summary_csv_path)

    # Population
    pop_north = pop[pop["region"].isin(["NE", "NW", "YH"])]
    pop_north = pop_north.groupby(["filename", "segmentation", "output code", "segment"], as_index=False)[["value"]].sum()
    pop_north["region"] = "The North"
    pop_north = pop_north[["filename", "segmentation", "output code", "region", "segment", "value"]]

    pop_eng = pop[pop["region"].isin(["NE", "NW", "YH",
                                      "EM", "WM", "EoE",
                                      "Lon", "SE", "SW"])]
    pop_eng = pop_eng.groupby(["filename", "segmentation", "output code", "segment"], as_index=False)[["value"]].sum()
    pop_eng["region"] = "England"
    pop_eng = pop_eng[["filename", "segmentation", "output code", "region", "segment", "value"]]

    pop_all = pop.groupby(["filename", "segmentation", "output code", "segment"], as_index=False)[["value"]].sum()
    pop_all["region"] = "GB"
    pop_all = pop_all[["filename", "segmentation", "output code", "region", "segment", "value"]]

    # Employment
    emp_north = emp[emp["region"].isin(["North East", "North West", "Yorkshire and The Humber"])]
    emp_north = emp_north.groupby(["filename", "segmentation", "output code", "segment"], as_index=False)[["value"]].sum()
    emp_north["region"] = "The North"
    emp_north = emp_north[["filename", "segmentation", "output code", "region", "segment", "value"]]

    emp_eng = emp[emp["region"].isin(["North East", "North West", "Yorkshire and The Humber",
                                      "East Midlands", "West Midlands", "East of England",
                                      "London", "South East", "South West"])]
    emp_eng = emp_eng.groupby(["filename", "segmentation", "output code", "segment"], as_index=False)[["value"]].sum()
    emp_eng["region"] = "England"
    emp_eng = emp_eng[["filename", "segmentation", "output code", "region", "segment", "value"]]

    emp_all = emp.groupby(["filename", "segmentation", "output code", "segment"], as_index=False)[["value"]].sum()
    emp_all["region"] = "GB"
    emp_all = emp_all[["filename", "segmentation", "output code", "region", "segment", "value"]]

    pop_output = pd.concat([pop_north, pop_eng, pop_all])
    emp_output = pd.concat([emp_north, emp_eng, emp_all])

    pop_output.to_csv(pop_output_csv_path)
    emp_output.to_csv(emp_output_csv_path)


summarise_population_outputs(
    output_file_name='population_forecast_output_summary_20250626',
    regions=constants.GORS + ["Scotland"],
    years_to_extract=[2033, 2038, 2043, 2048, 2053])

# summarise_pop_and_hh_outputs_multi_segments(
#     output_file_name="Pop_adults_children_summary_original_outputs_with_max_cap_95th_NW_2053",
#     segments_to_summarise=["adults", "children"],
#     years_to_summarise=[2053],
#     regions=["NW"],
#     output_file_prefixes=["Pop", "Households"],
#     forecast_pop_output_dir=Path(r"F:\Working\Land-Use\forecast_population_pop_test_20250605_cap\02_Final Outputs\max_cap_test_95th"),
#     output_dir=Path(r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\household_composition_growth")
# )

summarise_emp_outputs(
    output_file_name='employment_forecast_output_summary_20250519',
    years_to_extract=[2033, 2038, 2043, 2048, 2053])

summarise_household_outputs(
    output_file_name='household_forecast_output_summary_20250626',
    regions=constants.GORS + ["Scotland"],
    years_to_extract=[2023, 2033, 2038, 2043, 2048, 2053])

calculate_occupancies(
    forecast_years=[2033, 2038, 2043, 2048, 2053],
    regions=constants.GORS + ["Scotland"],
    base_pop_path=Path(fr"F:\Deliverables\Land-Use\2025-06 Release\Population\02_Final Outputs"),
    forecast_pop_path=Path(
        fr"F:\Working\Land-Use\forecast_population_20250626\02_Final Outputs"),
    agg_segments=["adults", "children"],
    output_file_name=f"occupancies_summary_adults_children_20250626")

# dvector_segment_comparisons(
#     dvector_dict={
#         "2023_LU_Base": r"F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs\Output E6.hdf",
#         "2043": r"F:\Working\Land-Use\temp_forecast_employment_testing_moving_to_config"
#         r"\01_Intermediate Files\Output Emp_SIC_1_digit_2043.hdf",
#     },
#     output_file_name=Path(
#         r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\test.csv"
#     ),
#     forecast_type="emp"
# )

combine_outputs_to_summary_regions(
    pop_summary_csv_path=POP_ANALYSIS_DIR / "population_forecast_output_summary_20250626.csv",
    emp_summary_csv_path=EMP_ANALYSIS_DIR / "employment_forecast_output_summary_20250519.csv",
    hh_summary_csv_path=POP_ANALYSIS_DIR / "household_forecast_output_summary_20250626.csv",
    pop_output_csv_path=POP_ANALYSIS_DIR / "population_forecast_output_summary_20250626_agg_rgns.csv",
    emp_output_csv_path=EMP_ANALYSIS_DIR / "employment_forecast_output_summary_20250519_agg_rgns.csv",
    hh_output_csv_path=POP_ANALYSIS_DIR / "household_forecast_output_summary_20250626_agg_rgns.csv"
)

# combine_base_to_summary_regions(
#     base_pop_summary_csv_path=Path(
#         r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\pop\Base_segment_totals_rgn.csv"),
#     base_emp_summary_csv_path=Path(
#         r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\emp\Base_emp_segment_totals_rgn.csv"),
#     pop_output_csv_path=Path(
#         r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\pop\Base_segment_totals_rgn_agg_rgns.csv"),
#     emp_output_csv_path=Path(
#         r"F:\Working\Land-Use\FORECASTING_analysis\Analysis\outputs\emp\Base_emp_segment_totals_rgn_agg_rgns.csv")
# )
