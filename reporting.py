from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from caf.base import ZoningSystem
import yaml

from land_use.constants.geographies import CACHE_FOLDER
from land_use.data_processing.mapping import create_interactive_maps
from land_use.data_processing import OutputLevel, translate_and_combine_dvectors, generate_segment_bar_plots
from land_use.reporting import templating

# TODO: expand on the documentation here
parser = ArgumentParser(description='Land-Use base population command line runner')
parser.add_argument('scenario_name', type=str, help='Name to use in output folder creation')
parser.add_argument('config_file', type=Path, nargs='+')
args = parser.parse_args()

scenario_name = args.scenario_name

# Set up the root results page
docs_dir = Path(__file__).parent / 'docs' / 'Scenario Results' / scenario_name
if not docs_dir.is_dir():
    docs_dir.mkdir(exist_ok=True, parents=True)
    with open(docs_dir / 'index.rst', 'w') as docs_index:
        docs_index.write(templating.render_scenario_page(scenario_name))

file_dict = defaultdict(list)

for cf in args.config_file:
    # load configuration file
    with open(cf, 'r') as text_file:
        config = yaml.load(text_file, yaml.SafeLoader)

    # Get output directory of main model outputs from config file
    OUTPUT_DIR = Path(config['output_directory']) / OutputLevel.FINAL

    # get files from existing output
    file_dict['Households'].extend(OUTPUT_DIR.glob('Output P11.1_*.hdf'))
    file_dict['Population'].extend(OUTPUT_DIR.glob('Output P13_*.hdf'))
    file_dict['Employment'].extend(OUTPUT_DIR.glob('Output E4.hdf'))

# define zone systems to translate to. NOTE: the map zone system must aggregate to the chart zone system if the two are different
MAP_ZONE_SYSTEM = 'LAD2023+SCOTLANDLAD'
CHART_ZONE_SYSTEM = 'RGN2021+SCOTLANDRGN'

# Calculate all of our "total" dictionaries in one go
data_dict = {}
for key, input_files in file_dict.items():
    print(key)
    if not input_files:
        continue
    
    data_dict[key] = translate_and_combine_dvectors(
        input_files=input_files,
        aggregate_zone_system=MAP_ZONE_SYSTEM
    )

if 'Population' in data_dict.keys():
    data_dict['Working Age Population'] = data_dict['Population'].filter_segment_value(
        'age_9', [4, 5, 6, 7, 8]
    )

for unit, map_total_dvector in data_dict.items():
    # Set up the output directory for that unit category
    unit_docs_dir = docs_dir / unit
    unit_docs_dir.mkdir(exist_ok=True)

    chart_total_dvector = map_total_dvector.translate_zoning(
        ZoningSystem.get_zoning(CHART_ZONE_SYSTEM, search_dir=CACHE_FOLDER)
    )

    # Store all segment names so we can figure out which ones we skip
    all_segment_names = set(chart_total_dvector.segmentation.names)

    # And set up the folder for all the results to go into
    results_dir = unit_docs_dir / 'Segment Results'
    results_dir.mkdir(exist_ok=True)

    for segment_plot in generate_segment_bar_plots(chart_total_dvector, unit=unit):
        # First - save the figure
        segment_plot.figure.savefig(results_dir / f'{segment_plot.segments}.png')

        # And save the data
        segment_plot.summary_data.to_csv(
            results_dir / f'{segment_plot.segments}.csv', 
            float_format=lambda x: '{:,.0f}'.format(x)
        )

        # Make the maps - note we filter to the northern regions here
        map_paths = create_interactive_maps(
            map_total_dvector, output_folder=results_dir, 
            specific_segment=segment_plot.segment_identifiers[0],
            filter_by={'RGN21CD': ['E12000001', 'E12000002', 'E12000003']},
            simplification=500
        )

        # Then fill out the template
        with open(results_dir / f'{segment_plot.segments}.rst', 'w') as segment_page:
            segment_page.write(
                templating.render_segment_page(
                    segment_name=segment_plot.segments,
                    graph_paths=[f'{segment_plot.segments}.png'],
                    table_paths=[f'{segment_plot.segments}.csv'],
                    map_paths=[p.name for p in map_paths]
                )
            )

        all_segment_names -= set(segment_plot.segment_identifiers)

    # Write the data type page now we know which segments have been skipped
    with open(unit_docs_dir / 'index.rst', 'w') as unit_index:
        unit_index.write(
            templating.render_data_type_page(
                data_type=unit,
                files_used=file_dict[unit],
                skipped_segments=all_segment_names
            )
        )
    