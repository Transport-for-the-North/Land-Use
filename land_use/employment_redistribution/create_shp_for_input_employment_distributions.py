from pathlib import Path

import geopandas as gpd
import pandas as pd

MAIN_DIR = Path(r"I:\NorMITs Land Use\2023\import\Employment Attraction Distributions")
DISTRIBUTION_DIRECTORY = Path(MAIN_DIR / "intermediate steps")
MAP_DIR = Path(MAIN_DIR / "map")


def create_shp_for_distributions(distribution_name: str) -> None:
    """Create a shapefile showing the distribution splits for LSOA/LAD

    Args:
        distribution_name (str): Descriptive name for distribution used in output
    """

    # TODO: consider if the 
    df = pd.read_csv(DISTRIBUTION_DIRECTORY / "lsoa_distributions_by_type.csv")

    MAP_DIR.mkdir(exist_ok=True)

    shp = gpd.read_file(MAP_DIR / "LSOA_2021_EnglandWales_NORTH.shp")

    df = df.rename(columns={"lsoa21cd": "lsoa"})

    shp = shp.rename(columns={"LSOA21CD": "lsoa"})

    # renmove columns that are duplicated in shp that we aren't joining on
    df = df.drop(columns=["lsoa21nm", "lad21cd", "rgn21cd", "rgn21nm"])

    shp_with_values = shp.merge(df)

    # shapefiles have restrictions on column widths, so doing some changes here
    for col in shp_with_values.columns:
        new_string = col.replace("_", "")
        new_string = new_string.replace("floorspace", "fs")
        new_string = new_string.replace("value", "vl") # ideally would have this as longer
        new_string = new_string.replace("rgnshortcode", "rgnshcd")
        new_string = new_string.replace("studentsallages", "students")
        new_string = new_string.replace("schoolpupils", "pupils")
        shp_with_values = shp_with_values.rename(columns={col: new_string})

    shp_with_values.to_file(
        MAP_DIR / f"lsoa_distribution_{distribution_name}.shp", driver="ESRI Shapefile"
    )


if __name__ == "__main__":
    create_shp_for_distributions(distribution_name="approach_a_weighting_2")
