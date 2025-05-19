from pathlib import Path

import pandas as pd


def extract_specific_segment_totals(filepath: Path, segment: str) -> pd.DataFrame:
    year = filepath.stem.split("_")[1]
    region = filepath.stem.split("_")[-1]
    df = pd.read_csv(filepath)
    soc_values = df[df["segmentation"] == segment].copy()
    soc_values["year"] = year
    soc_values["region"] = region
    return soc_values



if __name__ == "__main__":
    folder = Path(
        r"F:\Working\Land-Use\temp_forecast_employment_testing_moving_to_config_with_ons_pop_growth_all_sic\03_Assurance"
    )
    filepaths = folder.glob("Output Emp_*_segment_totals_*.csv")

    seg = "soc"

    df = pd.concat(
        [
            extract_specific_segment_totals(filepath=filepath, segment=seg)
            for filepath in filepaths
        ]
    )

    df.to_csv(folder / f"{seg}_segment_totals_all_years.csv", index=False)
