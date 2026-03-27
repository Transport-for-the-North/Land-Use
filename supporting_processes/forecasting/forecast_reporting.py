from pathlib import Path
from collections import defaultdict

from caf.base import ZoningSystem

from land_use.reporting import templating
from land_use.constants.geographies import CACHE_FOLDER
from land_use.data_processing import OutputLevel, translate_and_combine_dvectors, generate_segment_bar_plots

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use(r'https://raw.githubusercontent.com/Transport-for-the-North/caf.viz/main/src/caf/viz/tfn.mplstyle')

scenario_name = '2025-07 Issued Results'
docs_dir = Path(r'C:\OneDrive\OneDrive - AECOM\Code\Land-Use') / 'docs' / 'Scenario Results' / scenario_name
# Set up the root results page
if not docs_dir.is_dir():
    docs_dir.mkdir(exist_ok=True, parents=True)
    with open(docs_dir / 'index.rst', 'w') as docs_index:
        docs_index.write(templating.render_scenario_page(scenario_name))

# Get output directory of main model outputs from config file
POP_OUTPUT_DIR = Path(
    r'F:\Working\Land-Use\forecast_population_norcom_test\02_Final Outputs')
EMP_OUTPUT_DIR = Path(
    r'F:\Working\Land-Use\forecast_employment_20250717\02_Final Outputs')

region_mapping = {
    'W92000004': 'Wales',
    'E12000008': 'South East',
    'E12000004': 'East Midlands',
    'E12000005': 'West Midlands',
    'E12000002': 'North West',
    'E12000009': 'South West',
    'E12000007': 'London',
    'E12000003': 'Yorkshire and The Humber',
    'E12000001': 'North East',
    'E12000006': 'East of England',
    'S92000003': 'Scotland',
}


# Produce bar charts and tables
def produce_bar_charts_and_tables():
    for year in [2033, 2038, 2043, 2048, 2053]:
        file_dict = defaultdict(list)

        # get files from existing output
        file_dict['Population'].extend(POP_OUTPUT_DIR.glob(f'Output Pop_*{year}.hdf'))
        file_dict['Households'].extend(POP_OUTPUT_DIR.glob(f'Output Households_*_{year}.hdf'))

        file_dict['Employment'].extend(EMP_OUTPUT_DIR.glob(f'Output Emp_{year}.hdf'))

        # define zone systems to translate to
        CHART_ZONE_SYSTEM = 'RGN2021+SCOTLANDRGN'

        # Calculate all of our "total" dictionaries in one go
        data_dict = {}
        for key, input_files in file_dict.items():
            print(key)
            print(input_files)
            if not input_files:
                continue

            data_dict[key] = translate_and_combine_dvectors(
                input_files=input_files,
                aggregate_zone_system=CHART_ZONE_SYSTEM
            )

        if 'Population' in data_dict.keys():
            data_dict['Working Age Population'] = data_dict['Population'].filter_segment_value(
                'age_ntem', [2]
            )
            file_dict['Working Age Population'] = file_dict['Population']

        for unit, map_total_dvector in data_dict.items():
            # Set up the output directory for that unit category
            unit_docs_dir = docs_dir / unit
            unit_docs_dir.mkdir(exist_ok=True, parents=True)

            chart_total_dvector = map_total_dvector.translate_zoning(
                ZoningSystem.get_zoning(CHART_ZONE_SYSTEM, search_dir=CACHE_FOLDER)
            )

            # Store all segment names so we can figure out which ones we skip
            all_segment_names = set(chart_total_dvector.segmentation.names)

            # And set up the folder for all the results to go into
            results_dir = unit_docs_dir / 'Segment Results'
            results_dir.mkdir(exist_ok=True, parents=True)

            for segment_plot in generate_segment_bar_plots(chart_total_dvector, unit=unit):
                # First - save the figure
                segment_plot.figure.savefig(results_dir / f'{segment_plot.segments}_{year}.png')

                # And save the data
                segment_plot.summary_data.to_csv(
                    results_dir / f'{segment_plot.segments}_{year}.csv',
                    float_format=lambda x: '{:,.0f}'.format(x)
                )

                with open(results_dir / f'{segment_plot.segments}_{year}.rst', 'w') as segment_page:
                    segment_page.write(
                        templating.render_segment_page(
                            segment_name=f'{segment_plot.segments}_{year}',
                            graph_paths=[f'{segment_plot.segments}_{year}.png'],
                            table_paths=[f'{segment_plot.segments}_{year}.csv'],
                            map_paths=[]
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


def produce_total_line_charts(measure: str):
    """
    Produce totals summary of forecasts (by region), as line charts
    Parameters
    ----------
    measure: str
        E.g. Pop, Households, Emp

    -------

    """
    if measure == 'Pop':
        file_path = POP_OUTPUT_DIR
        unit_doc = 'Population'
    elif measure == 'Households':
        file_path = POP_OUTPUT_DIR
        unit_doc = 'Households'
    else:
        file_path = EMP_OUTPUT_DIR
        unit_doc = 'Employment'

    # Produce line charts across the years
    year_1 = defaultdict(list)
    year_2 = defaultdict(list)
    year_3 = defaultdict(list)
    year_4 = defaultdict(list)
    year_5 = defaultdict(list)

    # get files from existing output
    if measure == 'Pop' or measure == 'Households':
        year_1[unit_doc].extend(file_path.glob(f'Output {measure}_*_2033.hdf'))
        year_2[unit_doc].extend(file_path.glob(f'Output {measure}_*_2038.hdf'))
        year_3[unit_doc].extend(file_path.glob(f'Output {measure}_*_2043.hdf'))
        year_4[unit_doc].extend(file_path.glob(f'Output {measure}_*_2048.hdf'))
        year_5[unit_doc].extend(file_path.glob(f'Output {measure}_*_2053.hdf'))
    else:
        year_1[unit_doc].extend(file_path.glob(f'Output {measure}_2033.hdf'))
        year_2[unit_doc].extend(file_path.glob(f'Output {measure}_2038.hdf'))
        year_3[unit_doc].extend(file_path.glob(f'Output {measure}_2043.hdf'))
        year_4[unit_doc].extend(file_path.glob(f'Output {measure}_2048.hdf'))
        year_5[unit_doc].extend(file_path.glob(f'Output {measure}_2053.hdf'))

    # define zone systems to translate to
    CHART_ZONE_SYSTEM = 'RGN2021+SCOTLANDRGN'

    final_dfs = []
    for yr_files in year_1, year_2, year_3, year_4, year_5:
        data_dictionary = {}
        for key, input_files in yr_files.items():
            print(key)
            print(input_files)
            if not input_files:
                continue

            data_dictionary[key] = translate_and_combine_dvectors(
                input_files=input_files,
                aggregate_zone_system=CHART_ZONE_SYSTEM
            )

            if 'total' in data_dictionary[key].segmentation.names:
                data_dictionary[key] = data_dictionary[key].aggregate(segs=['total'])
            else:
                data_dictionary[key] = data_dictionary[key].add_segments(new_segs=['total']).aggregate(
                    segs=['total']
                )

            # save as a dataframe with new column defined as year
            df = data_dictionary[key].data
            if yr_files == year_1:
                yr = '2033'
            elif yr_files == year_2:
                yr = '2038'
            elif yr_files == year_3:
                yr = '2043'
            elif yr_files == year_4:
                yr = '2048'
            else:
                yr = '2053'

            df['year'] = yr

            # append out of the loop
            final_dfs.append(df)

    # concat together
    final_output = pd.concat(final_dfs)

    # map column titles to region names
    final_output = final_output.rename(columns=region_mapping)
    final_output['year'] = pd.to_numeric(final_output['year'])

    columns_to_plot = [col for col in final_output.columns if col != 'year']

    # plot figure
    fig, ax = plt.subplots(figsize=(10, 6))
    for x, column in enumerate(columns_to_plot):
        linestyle = '--' if x >= len(columns_to_plot) - 2 else '-'
        ax.plot(final_output['year'], final_output[column], marker='o', label=column, linestyle=linestyle)

    # set x-axis format
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    # set y-axis format
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    # set y-axis to begin at 0
    plt.ylim(bottom=0)

    plt.title(f'Total {unit_doc} Across Years')
    plt.xlabel('Year')
    plt.ylabel('Total')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set up the output directory for that unit category
    unit_docs_dir = docs_dir / unit_doc
    unit_docs_dir.mkdir(exist_ok=True, parents=True)

    results_dir = unit_docs_dir / 'Segment Results'
    results_dir.mkdir(exist_ok=True, parents=True)

    # First - save the figure
    plt.savefig(results_dir / f'total_all_years.png')

    # And save the data
    final_output_formatted = final_output.set_index('year').T.reset_index()
    final_output_formatted = final_output_formatted.rename(columns={'index': 'Region'})
    final_output_formatted.to_csv(
        results_dir / 'total_all_years.csv',
        float_format=lambda x: '{:,.0f}'.format(x)
    )

    with open(results_dir / f'total_all_years.rst', 'w') as segment_page:
        segment_page.write(
            templating.render_segment_page(
                segment_name='total_all_years',
                graph_paths=['total_all_years.png'],
                table_paths=['total_all_years.csv'],
                map_paths=[]
            )
        )


produce_bar_charts_and_tables()
produce_total_line_charts(measure='Pop')
produce_total_line_charts(measure='Households')
produce_total_line_charts(measure='Emp')
