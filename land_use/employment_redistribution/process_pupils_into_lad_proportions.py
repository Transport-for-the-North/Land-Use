from pathlib import Path

import numpy as np
import pandas as pd


MAIN_DIR = Path(r"I:\NorMITs Land Use\2023\import\Employment Attraction Distributions")
INPUT_DIR = MAIN_DIR / "raw data"
OUTPUT_DIR = MAIN_DIR / "intermediate steps"

GEO_LU = Path(
    r"F:\Working\Land-Use\SHAPEFILES\CORRESPONDENCES\EW_LSOA21_TO_MSOA21_TO_LAD21_TO_GOR21.csv"
)


def main():
    df = pd.read_csv(
        Path(INPUT_DIR, "spc_school_level_underlying_data_lsoa21_202223.csv")
    )

    lsoa_to_lad_lu = pd.read_csv(GEO_LU, usecols=["LSOA21CD", "LAD21CD"])

    lsoa_to_lad_lu = lsoa_to_lad_lu.rename(columns={"LSOA21CD": "lsoa21cd"})

    df_with_lad = pd.merge(df, lsoa_to_lad_lu)

    df_with_lad["lad_prop"] = df_with_lad.groupby("LAD21CD")["fte pupils"].transform(
        lambda x: x / x.sum()
    )

    df_with_lad = df_with_lad[["lsoa21cd", "lad_prop"]]

    # values are only given for England and where we have them so need to 
    # distinguish between non English LSOAs and English LSOAs with no pupiils

    all_zones = []
    with open(INPUT_DIR / "all_lsoa_dzs.txt", "r") as f:
        for line in f:
            all_zones.append(line.strip())

    all_zones_df = pd.DataFrame({"lsoa21cd": all_zones})

    df_with_lad = pd.merge(all_zones_df, df_with_lad, how="left")

    df_with_lad["lad_prop"] = np.where(
        (
            df_with_lad["lsoa21cd"].str.startswith("E")
            & df_with_lad["lad_prop"].isna()
        ),
        0.0,
        df_with_lad["lad_prop"],
    )
    
    output_df = df_with_lad[["lsoa21cd", "lad_prop"]]

    output_df.to_csv(OUTPUT_DIR / "fte_pupil_proportions.csv", index=False)


if __name__ == "__main__":
    main()
