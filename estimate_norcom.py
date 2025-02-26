from argparse import ArgumentParser
from pathlib import Path
import shutil

import yaml
from caf.brain.ml.main_models.prediction_model.__main__ import main
from caf.brain.ml.main_models.prediction_model.prediction_model_inputs import run_file_inputs, Models

# TODO: expand on the documentation here
parser = ArgumentParser('NorCOM parameter estimation command line runner')
parser.add_argument('config_file', type=Path)
args = parser.parse_args()

# load configuration file
with open(args.config_file, 'r') as text_file:
    config = yaml.load(text_file, yaml.SafeLoader)

# get original output folder
OUTPUT_DIR = Path(config['output_path'])
# copy config for traceability
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
shutil.copy(
    args.config_file,
    OUTPUT_DIR / args.config_file.name
)

# set up model inputs
if isinstance(config['model_choice'], str):
    config['model_choice'] = [Models[config['model_choice']]]
else:
    config['model_choice'] = [Models[model] for model in config['model_choice']]

# set up params and run
params = run_file_inputs(**config)
try:
    main(params)
except Exception as e:
    with open(fr'{OUTPUT_DIR}\output\errors.log', 'w') as error_file:
        error_file.write(str(e))
    print(e)
