from pathlib import Path

from caf.base.data_structures import DVector


def main(run_dir: Path) -> None:
    dvect_dir = run_dir / "02_Final Outputs"
    assurance_dir = run_dir / "03_Assurance"

    assurance_dir.mkdir(exist_ok=True)

    print("loading e4")
    e4 = DVector.load(dvect_dir / "Output E4.hdf")
    calculate_totals(dv=e4, name="e4", out_dir=assurance_dir)
    print("loading e4.3")
    e4_3 = DVector.load(dvect_dir / "Output E4_3.hdf")
    calculate_totals(dv=e4_3, name="e4_3", out_dir=assurance_dir)
    print("loading e5")
    e5 = DVector.load(dvect_dir / "Output E5.hdf")
    calculate_totals(dv=e5, name="e5", out_dir=assurance_dir)
    print("loading e6")
    e6 = DVector.load(dvect_dir / "Output E6.hdf")
    calculate_totals(dv=e6, name="e6", out_dir=assurance_dir)


def calculate_totals(dv: DVector, name: str, out_dir: Path) -> None:

    dv_index = dv.data.index.names

    if "soc" in dv_index:
        soc = dv.aggregate(["soc"])
        soc_transposed = soc.data.transpose()
        soc_transposed.to_csv(out_dir / f"{name}_soc_transposed.csv")

    if "sic_1_digit" in dv_index:
        sic = dv.aggregate(["sic_1_digit"])
        sic_transposed = sic.data.transpose()
        sic_transposed.to_csv(out_dir / f"{name}_sic_transposed.csv")

        sic_totals = sic_transposed.sum().reset_index()
        sic_totals.to_csv(out_dir / f"{name}_sic_totals.csv")
    
    # note other segmnentations probably contain too many values to be useful


if __name__ == "__main__":
    run_dir = Path(
        r"F:\Working\Land-Use\OUTPUTS_base_employment_bres_approach_a_weighting_2_level_check"
    )
    main(run_dir=run_dir)
