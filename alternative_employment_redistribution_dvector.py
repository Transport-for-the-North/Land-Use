from pathlib import Path
import yaml

import numpy as np
import pandas as pd

import land_use.preprocessing as pp


EMPLOYMENT_DISTRIBUTION_DIR = Path(
    r"I:\NorMITs Land Use\2023\import\Employment Attraction Distributions"
)
RAW_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "raw data"
INTERMIDIATE_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "intermediate steps"
DVECTOR_DIR = EMPLOYMENT_DISTRIBUTION_DIR / "sic mapped distributions"


# So implied figures for m2 / fte
INDUSTRY = 36
OFFICE = 12
OTHER = 70
RETAIL = 19


def main():
    create_table_of_entries_and_proortions()

    yaml_path = (
        EMPLOYMENT_DISTRIBUTION_DIR
        / "adjusting_employment_new_distribution_approach_a_weighting_2.yml"
    )
    create_lsoa_sic_factors_dvector(yaml_path=yaml_path)


def create_table_of_entries_and_proortions():
    voa_df = pd.read_csv(
        INTERMIDIATE_DIR / "infilled_voa_floorspace_rate_and_values.csv"
    )

    voa_df = voa_df.rename(columns={"lsoa_code": "lsoa21cd"})

    voa_df["voa_jobs"] = (
        voa_df["industry_floorspace"] * INDUSTRY
        + voa_df["office_floorspace"] * OFFICE
        + voa_df["other_floorspace"] * OTHER
        + voa_df["retail_floorspace"] * RETAIL
    )

    education_df = pd.read_csv(Path(RAW_DIR, "pupils_fe_he_111524.csv"))
    education_df = education_df.rename(
        columns={
            "fe": "futher_education",
            "he": "higher_education",
            "total": "students_all_ages",
        }
    )

    combined_df = pd.merge(voa_df, education_df, how="outer")
    combined_df = combined_df.fillna(0)

    lsoa_to_rgn_lu = pd.read_csv(EMPLOYMENT_DISTRIBUTION_DIR / "lsoa_to_rgn_ews.csv")
    combined_df = pd.merge(combined_df, lsoa_to_rgn_lu)
    combined_df.to_csv(INTERMIDIATE_DIR / "distribution_values.csv", index=False)

    # and now work out proportions
    proportions_df = combined_df.copy()

    numeric_cols = proportions_df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        proportions_df[col] = proportions_df.groupby(["lad_code"])[col].transform(
            lambda x: x / x.sum()
        )

    proportions_df.to_csv(
        INTERMIDIATE_DIR / "new_distribution_proportions.csv", index=False
    )


def create_lsoa_sic_factors_dvector(yaml_path: Path) -> None:

    with open(yaml_path) as text_file:
        config = yaml.load(text_file, yaml.SafeLoader)

    rgn_to_adj = config["rgn_to_adj"]

    distribution_approach = config["distribution_approach"]
    distribution_input = config["distribution_input"]

    lsoa_type_distributions = pd.read_csv(
        INTERMIDIATE_DIR / distribution_input
    )

    lsoa_type_distributions = lsoa_type_distributions[
        lsoa_type_distributions["rgn_short_code"].isin(rgn_to_adj)
    ]

    sic_allocation = pd.DataFrame(
        list(config["sic_section_type_adj"].items()),
        columns=["sic_1_digit", "distribution type"],
    )

    distribution_by_type_wide = lsoa_type_distributions.transpose()
    distribution_by_type_wide.columns = distribution_by_type_wide.loc["lsoa21cd"]
    distribution_by_type_wide = distribution_by_type_wide.reset_index(
        names="distribution type"
    )

    distribution_by_type_wide = distribution_by_type_wide[
        distribution_by_type_wide["distribution type"] != "lsoa21cd"
    ]

    # checking requested distribtions are available
    available_distributions = set(
        distribution_by_type_wide["distribution type"].to_list()
    )

    available_distributions.add("none")

    requested_distrubtions = set(sic_allocation["distribution type"].to_list())

    missing_distributions = requested_distrubtions - available_distributions

    if missing_distributions:
        raise ValueError(
            f"The following distributions were requested but not available: {missing_distributions}\n"
            f"Available distributions are: {available_distributions}"
        )

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

    factors_out_path = DVECTOR_DIR / f"new_factors_{distribution_approach}.csv"

    DVECTOR_DIR.mkdir(exist_ok=True, parents=True)

    pp.save_preprocessed_hdf(source_file_path=factors_out_path, df=factors)

    # Also write as a transposed csv for ease of checking
    factors.transpose().to_csv(
        DVECTOR_DIR / f"transposed_new_factors_{distribution_approach}.csv"
    )


if __name__ == "__main__":
    main()
