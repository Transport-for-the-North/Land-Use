from pathlib import Path
import yaml

import numpy as np
import pandas as pd

import land_use.preprocessing as pp
import furnessing_voa_floorspace_values

INPUT_DIR = Path(r"I:\NorMITs Land Use\2023\import")

EMPLOYMENT_DISTRIBUTION_DIR = INPUT_DIR / "Employment Attraction Distributions"
RAW_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "raw data"
INTERMIDIATE_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "intermediate steps"
DVECTOR_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "sic mapped distributions"


# So implied figures for m2 / fte
OFFICE = 12
INDUSTRY = 36
RETAIL = 19
OTHER = 70

# little bit of prework for ons lookup that is used extensively
ONS_LU = pd.read_csv(
    INPUT_DIR
    / "ONS"
    / "Correspondence_lists"
    / "EW_LSOA21_TO_MSOA21_TO_LAD21_TO_GOR21.csv"
)
ONS_LU.columns = map(str.lower, ONS_LU.columns)


def main():
    #furnessing_voa_floorspace_values.main()
    lsoa_type_distributions = create_lsoa_distributions_by_measure()
    create_lsoa_sic_factors(lsoa_type_distributions=lsoa_type_distributions)


def create_lsoa_distributions_by_measure() -> pd.DataFrame:

    # Jobs could be uplifted to take account of part-time/full-time but as we only
    # have this at LAD level, and only looking to redistribute within LAD this would
    # have no impact.
    floorspaces = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_lsoa_voa_floorspace.csv",
        usecols=["ons_name", "floor_type", "msoa", "current_value_updated"],
    )

    floorspaces = floorspaces.rename(
        columns={"ons_name": "lsoa21nm", "current_value_updated": "floorspace"}
    )

    lsoa_voa_jobs = calc_voa_jobs_splits(df=floorspaces)
    voa_floorspace_splits = calc_floorspace_splits(df=floorspaces)
    voa_floorspace_w_jobs = pd.merge(voa_floorspace_splits, lsoa_voa_jobs)

    lsoa_pupil_splits = calc_pupil_lsoa_splits()

    lsoa_type_distrib = pd.merge(lsoa_pupil_splits, voa_floorspace_w_jobs, how="outer")

    ons_lsoa_to_lad = ONS_LU[["lsoa21nm", "lad21cd"]]

    lsoa_type_distrib = pd.merge(lsoa_type_distrib, ons_lsoa_to_lad, how="outer")

    lsoa_type_distrib = prepare_for_export(df=lsoa_type_distrib)

    lsoa_type_distrib.to_csv(
        INTERMIDIATE_DIR / "lsoa_distributions_by_type.csv", index=False
    )
    return lsoa_type_distrib


def calc_floorspace_splits(df):

    df = pd.merge(df, ONS_LU[["lsoa21nm", "lad21cd", "rgn21nm"]])

    # this needs to be grouped by LAD not msoa (which requires adding lad column to the dataset)
    df["jobs_proportion"] = df.groupby(["lad21cd", "floor_type"])[
        "floorspace"
    ].transform(lambda x: x / x.sum())

    df_wide = df.pivot_table(
        index="lsoa21nm", columns="floor_type", values="jobs_proportion", fill_value=0
    ).reset_index()

    return df_wide


def calc_pupil_lsoa_splits() -> pd.DataFrame:

    df = pd.read_csv(
        Path(RAW_DIR, "spc_school_level_underlying_data_lsoa21_202223.csv")
    )

    df_with_lad = pd.merge(df, ONS_LU[["lsoa21cd", "lad21cd"]])

    df_with_lad["pupils"] = df_with_lad.groupby("lad21cd")["fte pupils"].transform(
        lambda x: x / x.sum()
    )

    # infill nas with 0, note at the moment this puts Wales and Scotland to 0 as well.
    df_with_lad["pupils"] = df_with_lad["pupils"].fillna(0.0)

    pupils_splits = df_with_lad[["lsoa21cd", "pupils"]]

    pupils_splits.to_csv(INTERMIDIATE_DIR / "fte_pupil_proportions.csv", index=False)

    lsoa_nm_cd_lu = ONS_LU[["lsoa21cd", "lsoa21nm", "rgn21cd", "rgn21nm"]]

    return pd.merge(pupils_splits, lsoa_nm_cd_lu, how="outer")


def calc_voa_jobs_splits(df: pd.DataFrame) -> pd.DataFrame:
    # calculate jobs proportions
    df_floorspace_to_jobs = pd.DataFrame(
        data={
            "floor_type": ["retail", "industry", "office", "other"],
            "density": [RETAIL, INDUSTRY, OFFICE, OTHER],
        }
    )
    df = pd.merge(df, df_floorspace_to_jobs)

    df["jobs"] = df["floorspace"] * df["density"]

    df = pd.merge(df, ONS_LU[["lsoa21nm", "lad21cd"]], how="outer")

    lsoa_prop_voa_jobs = (
        df.groupby(["lsoa21nm", "lad21cd"]).agg({"jobs": "sum"}).reset_index()
    )

    lsoa_prop_voa_jobs["jobs_proportion"] = lsoa_prop_voa_jobs.groupby("lad21cd")[
        "jobs"
    ].transform(lambda x: x / x.sum())

    lsoa_prop_voa_jobs = lsoa_prop_voa_jobs.drop(columns=["jobs", "lad21cd"])

    return lsoa_prop_voa_jobs


def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"jobs_proportion": "voa jobs"})

    rgn_short_code_lu = pd.DataFrame.from_records(
        [
            ("East of England", "EoE"),
            ("East Midlands", "EM"),
            ("London", "Lon"),
            ("North East", "NE"),
            ("North West", "NW"),
            ("Scotland", "Scotland"),
            ("South East", "SE"),
            ("South West", "SW"),
            ("West Midlands", "WM"),
            ("Wales", "Wales"),
            ("Yorkshire and The Humber", "YH"),
        ],
        columns=["rgn21nm", "rgn_short_code"],
    )

    df = pd.merge(df, rgn_short_code_lu)

    df = df.fillna(0.0)

    return df[
        [
            "lsoa21cd",
            "lsoa21nm",
            "lad21cd",
            "rgn21cd",
            "rgn21nm",
            "rgn_short_code",
            "voa jobs",
            "industry",
            "office",
            "retail",
            "other",
            "pupils",
        ]
    ]


def create_lsoa_sic_factors(lsoa_type_distributions: pd.DataFrame) -> None:
    yaml_path = EMPLOYMENT_DISTRIBUTION_DIR / "adjusting_employment_distribution.yml"
    with open(yaml_path) as text_file:
        config = yaml.load(text_file, yaml.SafeLoader)

    rgn_to_adj = config["rgn_to_adj"]

    lsoa_type_distributions = lsoa_type_distributions[
        lsoa_type_distributions["rgn_short_code"].isin(rgn_to_adj)
    ]
    sic_allocation = pd.DataFrame(
        list(config["sic_section_type_adj"].items()),
        columns=["sic_1_digit", "distribution type"],
    )

    lsoa_type_distributions = lsoa_type_distributions[
        ["lsoa21cd", "voa jobs", "industry", "office", "retail", "other", "pupils"]
    ]

    distribution_by_type_wide = lsoa_type_distributions.transpose()
    distribution_by_type_wide.columns = distribution_by_type_wide.loc["lsoa21cd"]
    distribution_by_type_wide = distribution_by_type_wide.reset_index(
        names="distribution type"
    )

    distribution_by_type_wide = distribution_by_type_wide[
        distribution_by_type_wide["distribution type"] != "lsoa21cd"
    ]

    factors = pd.merge(sic_allocation, distribution_by_type_wide, how="left")
    factors = factors.drop(columns="distribution type")

    # infill with nas for lsoas (and DZ) not included
    lsoa_dz_list = pd.read_csv(RAW_DIR / "all_lsoa_dzs.txt", names=["id"])[
        "id"
    ].to_list()

    missing_zones = list(set(lsoa_dz_list) - set(factors.columns))
    dummy_values = pd.DataFrame(
        np.NaN, index=factors["sic_1_digit"], columns=missing_zones
    )

    dummy_values = dummy_values.reset_index(names="sic_1_digit")

    factors = pd.merge(factors, dummy_values)
    factors = factors.set_index("sic_1_digit")

    factors_out_path = DVECTOR_DIR / "factors.csv"
    factors.transpose().to_csv(DVECTOR_DIR / "transposed_factors.csv")

    pp.save_preprocessed_hdf(source_file_path=factors_out_path, df=factors)

    # Also write as a transposed csv for ease of checking
    factors.to_csv(DVECTOR_DIR / "factors.csv")


if __name__ == "__main__":
    main()
