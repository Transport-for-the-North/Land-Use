# Expolotary Employment Configs

## Introduction

This directory consists of employment configuration files that are not used for
the final (deliverable run).

Early on in the project there was a discussion as to if the employment data should
be based on HSE/HSL or BRES data. BRES was used as an interim measure but then
as the HSE data became available it was decided to stick with BRES data.
This was due to a feeling that BRES better represented expected employment patterns.

## VOA adjustments

When looking at the original outputs it was felt by TfN that for some job types the distribution provided by BRES did not match expectation. So, adjustments were required. Currently this consists of shifting values within LADs, with the LAD total (by SIC) remaining constant.
There were two primary sources for this redistribution: VOA and Education numbers. VOA is provided at various geographies (including LAD, MSOA and LSOA), by different uses (all, industry, office, other, retail), and for floorspace and a rate.
Some values are masked due to small sample size (business < 5). And rates are not provided where there are no business of that type.

## Original Employment (furnessed) redistribution hdf

A furnessing/balancing process was used to fill in the VOA masked data, using the provided data as constraints. Noting that no data was masked at the LAD level, some was at the MSOA level, and a significant of cells were at the LSOA level.

To obtain a value, the floorspace was multiplied by the rate.

Scripts to preform this are:

* pre_processing_voa_inputs.py – deals with the masking and voa data extraction
* create_employment_redistribution_dvector.py – reads in the processed voa and education values, calculates the distribution by type (industry value, retail value, voa jobs, pupils,….). Creates the hdf required by base_employment by using this data and a yaml.

### Alternative Employment (infill from above) redistribution hdf

To test the sensitivity of the balancing process an alternative approach was also used for voa. This had a much simpler way to filling in the data, by using the geography above for floorspace proportions and VOA rates (e.g., LSOA was filled in using MSOA where needed, and LAD if MSOA was also masked).
The scripts used for this are alternative_employment_redistribution_dvector.py and alternative_preprocess_voa_values.py.

The yamls used a very similar apart from there is now an option to point to different distribution inputs “distribution_input”, so this way you can easily switch between the two distribution options (and potentially more in future).

On analysing the outputs the results were very similar and the original furnessed
approach was taken forward.

## Weightings

Two alternative weightings were tried. For both weightings the change was restricted to just those LADs in 'The North' (consisting of the three Regions: North East, North West, Yorkshire and The Humber).

Weighting 2 consists of the following changes by sic section (sic 1 digit):

* 3: industry_value
* 7: retail_value
* 9: other_value
* 10-15: office_value
* 16: Students_all_ages
* 17-19: other values
* Rest: none

Weighting 3 consists of similar to weighting 2:

* 3: industry_value
* 7: retail_value
* 10-15: office_value
* Rest: none

With the differences being for 9, 16, and 17-19.

It was decided to go with weighting 3, furnessed approach for the final output.
