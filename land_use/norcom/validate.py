from dataclasses import dataclass
from pathlib import Path


@dataclass
class Params:
    estimation_version: str
    results_path: Path
    zonal_lookups: Path
    input_dvectors: Path
    output_path: Path
    validation_dvector: Path = None

    @classmethod
    def from_yaml(cls, config):
        try:
            validation_dvector = Path(config['validation_dvector'])
        except KeyError:
            validation_dvector = None

        return Params(
            estimation_version=str(config['estimation_version']),
            results_path=Path(config['results_path']),
            zonal_lookups=Path(config['zonal_lookups']),
            input_dvectors=Path(config['input_dvectors']),
            validation_dvector=validation_dvector,
            output_path=Path(config['output_path']),
        )

    def validate(self) -> bool:
        return isinstance(self.validation_dvector, Path)
