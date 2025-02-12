from pathlib import Path

import pandas as pd


from caf.base.data_structures import DVector
from caf.base.zoning import TranslationWeighting
from land_use import constants


# map to LAs
# Keep Liverpool, Manchester and Bradford

# Employment
# for employment will read in all UK (E4 and E4_2)
# For employment need soc totals

# population
# only need NW and YH for population (P11 and P13.3)
# p11 will give age_9, soc
# p13.3 will provide accom_h, car_availability


LA_CODES_TO_KEEP = {
    "E08000032": "Bradford",
    "E08000012": "Liverpool",
    "E08000003": "Manchester",
}


EMPLOYMENT_DIR = Path(r"F:\Deliverables\Land-Use\241213_Employment\02_Final Outputs")
POPULATION_DIR = Path(r"F:\Deliverables\Land-Use\241220_Populationv2\02_Final Outputs")

OUTPUT_DIR = Path(r"F:\Working\Land-Use\FORECASTING_prep\01_Intermediate Files")


def main() -> None:
    employment()
    population()


def employment() -> None:

    dv_path = EMPLOYMENT_DIR / "Output E4.hdf"
    df = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["soc"])
    df.to_csv(Path(OUTPUT_DIR, "select_las_e4_soc.csv"))

    dv_path = EMPLOYMENT_DIR / "Output E4_2.hdf"
    df = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["soc"])
    df.to_csv(Path(OUTPUT_DIR, "select_las_e4_2_soc.csv"))


def population() -> None:
    p11_extraction()
    p13_extraction()


def p11_extraction() -> None:
    """Extract age_9 and soc totals from P13.3 for select LAs"""
    dv_path = POPULATION_DIR / "Output P11_YH.hdf"
    yh_age_9 = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["age_9"])
    yh_soc = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["soc"])

    dv_path = POPULATION_DIR / "Output P11_NW.hdf"
    nw_age_9 = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["age_9"])
    nw_soc = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["soc"])

    age_9 = pd.merge(yh_age_9.reset_index(), nw_age_9.reset_index())
    age_9.to_csv(Path(OUTPUT_DIR, "select_las_p11_age_9.csv"), index=False)

    soc = pd.merge(yh_soc.reset_index(), nw_soc.reset_index())
    soc.to_csv(Path(OUTPUT_DIR, "select_las_p11_soc.csv"), index=False)


def p13_extraction() -> None:
    """Extract accom_h and car_availability totals from P13.3 for select LAs"""
    dv_path = POPULATION_DIR / "Output P13.3_YH.hdf"
    yh_accom = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["accom_h"])
    yh_car_avail = extract_specific_la_totals_for_segs(
        dv_path=dv_path, segs=["car_availability"]
    )
    dv_path = POPULATION_DIR / "Output P13.3_NW.hdf"
    nw_accom = extract_specific_la_totals_for_segs(dv_path=dv_path, segs=["accom_h"])
    nw_car_avail = extract_specific_la_totals_for_segs(
        dv_path=dv_path, segs=["car_availability"]
    )

    accom_h = pd.merge(yh_accom.reset_index(), nw_accom.reset_index())
    accom_h.to_csv(Path(OUTPUT_DIR, "select_las_p13_accom_h.csv"), index=False)

    car_avail = pd.merge(yh_car_avail.reset_index(), nw_car_avail.reset_index())
    car_avail.to_csv(
        Path(OUTPUT_DIR, "select_las_p13_car_availability.csv"), index=False
    )


def extract_specific_la_totals_for_segs(dv_path: Path, segs: list[str]) -> pd.DataFrame:
    dv = DVector.load(dv_path)
    dv_lad = dv.translate_zoning(
        new_zoning=constants.LAD_EWS_2023_ZONING_SYSTEM,
        cache_path=constants.CACHE_FOLDER,
        weighting=TranslationWeighting.SPATIAL,
        check_totals=True,
    )

    dv_lad = dv_lad.aggregate(segs=segs)

    df_lad = dv_lad.data

    df_lad = df_lad[[c for c in df_lad.columns if c in LA_CODES_TO_KEEP.keys()]]

    df_lad = df_lad.rename(
        columns=dict(zip(LA_CODES_TO_KEEP.keys(), LA_CODES_TO_KEEP.values()))
    )

    return df_lad


if __name__ == "__main__":
    main()
