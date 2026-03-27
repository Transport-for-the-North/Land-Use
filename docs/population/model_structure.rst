Model Structure
###############

Introduction
============
This page can be used as a reference for the general flow of data within each of the models.

Base Year Population
====================

**Work in progress**

.. graphviz::

    digraph G {
        rankdir="LR"
        nodesep=0.5
        node [shape=record, color=blue width=3.4]
            communal_establishments [label="Census|Total population in \nCommunal Establishments|LSOA"]
            ce_type [label="Census|Total population in Communal \nEstablishments by CE type|MSOA"]
            ce_pop_soc [label="Census|Total population in Communal \nEstablishments by age, gender, SOC|GOR"]
            ce_pop_econ [label="Census|Total population in Communal \nEstablishments by age, gender, \neconomic status|GOR"]
            
            addressbase [label="Census 2021|Number of dwellings|LSOA"];
            age_gender [label="Census|Proportion of population by\ndwelling type, age, gender|MSOA"]

            occupied [label="Census|Number of occupied households|LSOA"]
            unoccupied [label="Census|Number of unoccupied households|LSOA"]

            table_1 [label="ONS Table 1|Population by dwelling type|LSOA"];
            table_2 [label="ONS Table 2|Proportion of households by\ndwelling type, #adults, #children, #cars|MSOA"];
            table_3 [label="ONS Table 3|Proportion of population by\neconomic status/employment status/SOC,\ndwelling type, NS-SeC|MSOA"];
            table_4 [label="ONS Table 4|Proportion of households by\ndwelling type, NS-SeC|LSOA"];

            ce_output [label="Communal Establishments|Population by CE type, age, gender, \neconomic status, SOC|LSOA"];

            mype_2022 [label="MYPE 2022|Population by age, gender|LSOA"]
            mype_2023 [label="MYPE 2023|Population by age, gender|LAD"]
            aps_20241 [label="APS 2024|Population by economic status, gender|GOR"]
            aps_20242 [label="APS 2024|Population by employment status, gender|GOR"]

            ons_2023 [label="ONS 2023|Households|LAD"]

            pop_constraint_1 [label="0 Households when 0 Population"]
            pop_constraint_2 [label="Maximum Household Occupancy by hh NS-SeC\nhh#adults, hh#children, hh#cars"]
            pop_constraint_3 [label="Minimum Household Occupancy by hh NS-SeC\nhh#adults, hh#children, hh#cars"]


        node [style=rounded, color=black]

            output_p1_1 [label="Output P1.1|Occupied Households|LSOA"];
            output_p1_2 [label="Output P1.2|Unoccupied Households|LSOA"];
            output_p1_3 [label="Output P1.3|Average Household Occupancy|LSOA"];
            output_p1_4 [label="Output P1.4|Non-Empty Proportion of Households|LSOA"];
            output_p1_5 [label="Output P1.5|Unnoccupied Factor|LSOA"];
            output_p2 [label="Output P2|Adjusted Number of Dwellings|LSOA"];
            output_p3 [label="Output P3|Households by dwelling type, NS-Sec|LSOA"];
            output_p4_1 [label="Output P4.1|Households by dwelling type, NS-SeC\n#adults, #children, #cars|LSOA"];
            output_p4_2 [label="Output P4.2|Households rebalanced with input datasets|LSOA"];
            output_p4_3 [label="Output P4.3|Households rebalanced with independent datasets|LSOA"];
            output_p5 [label="Output P5|Population by dwelling type, NS-SeC\nhh#adults, hh#children, hh#cars|LSOA"];
            output_p6 [label="Output P6|Population by dwelling type, NS-SeC\nhh#adults, hh#children, hh#cars,\nage, gender|LSOA"];
            output_p7 [label="Output P7|Population by dwelling type, NS-SeC\nhh#adults, hh#children, hh#cars,\nage, gender, economic status,\nemployment status, SOC|LSOA"];
            output_p8 [label="Output P8|Population by dwelling type, NS-SeC\nhh#adults, hh#children, hh#cars,\nage, gender, economic status,\nemployment status, SOC|LSOA"];
            output_p9 [label="Output P9|Population rebalanced with input datasets|LSOA"];
            output_p10 [label="Output P10|Population rebalanced with independent datasets|LSOA"];

            output_p11 [label="Output P11|2023 Population by dwelling type, NS-SeC\nhh#adults, hh#children, hh#cars,\nage, gender, economic status,\nemployment status, SOC|LSOA"];
            output_p12 [label="Output P12|2023 Households by dwelling type, NS-SeC\n#adults, #children, #cars|LSOA"];
            output_p13_1 [label="Output P13.1|2023 Households by dwelling type, NS-SeC\n#adults, #children, #cars|LSOA"];
            output_p13_2 [label="Output P13.2|2023 Households by dwelling type, NS-SeC\n#adults, #children, #cars|LSOA"];
            output_p13_3 [label="Output P13.3|2023 Households by dwelling type, NS-SeC\n#adults, #children, #cars|LSOA"];
            output_p14_1 [label="Output P14.1|2023 Occupied Households by dwelling type|LSOA"];
            output_p14_2 [label="Output P14.2|2023 Unoccupied Households by dwelling type|LSOA"];

        occupied -> output_p1_1;
        unoccupied -> output_p1_2;
        occupied -> output_p1_4;
        unoccupied -> output_p1_4;
        occupied -> output_p1_5;
        unoccupied -> output_p1_5;

        occupied -> output_p1_3;
        unoccupied -> output_p1_3;
        table_1 -> output_p1_3;

        output_p1_1 -> output_p2;
        output_p1_2 -> output_p2;
        addressbase -> output_p2

        table_4 -> output_p3;
        output_p2 -> output_p3;
        output_p3 -> output_p4_1;
        table_2 -> output_p4_1

        output_p4_1 -> output_p4_2;
        table_2 -> output_p4_2;
        output_p4_2 -> output_p4_3

        output_p4_3 -> output_p5;
        output_p1_3 -> output_p5

        age_gender -> output_p6;
        output_p5 -> output_p6

        table_3 -> output_p7;
        output_p6 -> output_p7

        communal_establishments -> ce_output
        ce_type -> ce_output
        ce_pop_soc -> ce_output
        ce_pop_econ -> ce_output
        ce_output -> output_p8;
        output_p7 -> output_p8

        output_p8 -> output_p9
        age_gender -> output_p9

        output_p9 -> output_p10

        output_p10 -> output_p11
        mype_2022 -> output_p11
        mype_2023 -> output_p11
        aps_20241 -> output_p11
        aps_20242 -> output_p11

        output_p4_3 -> output_p12
        output_p11 -> output_p12
        ons_2023 -> output_p12

        output_p12 -> output_p13_1
        pop_constraint_1 -> output_p13_1

        output_p13_1 -> output_p13_2
        pop_constraint_2 -> output_p13_2

        output_p13_2 -> output_p13_3
        pop_constraint_3 -> output_p13_3

        output_p13_3 -> output_p14_1
        output_p14_1 -> output_p14_2
        output_p1_5 -> output_p14_2

    }

Forecast Population
===================

The forecasting targets have been derived from the data sources previously defined. Growth factors have been calculated
based on the datasets and applied to the Land-Use Base DVectors, to derive targets.

As the ONS regional population projections only cover up to 2043 at the time of developing the forecasts, the growth
targets from 2043 to 2053 have been extrapolated, using national growth projections, from the final year of 2043.

As the Labour Market & Skills data only projects up to 2035, this data has been extrapolated for use in the forecasts.
For this, the SIC 1 digit / SOC segment splits were taken from the final year of 2035 and then the population growth
from 2035 onwards has been used to align the jobs growth with the total population. This maintains the same jobs per
resident ratio at the regional level.

The population targets are calculated prior to the main forecast process, in the following script:

- reformat_forecast_population_input_data.py

The target hdf files contain the forecast years required - additional years can be calculated and added as as separate
key within the hdf.

The config files are set up to allow various input targets, as well as the option to maintain Base Land-Use
distributions for specific segments. This allows for flexibility and additional targets to be processed and added in
future.

.. graphviz::

    digraph G {
        rankdir="LR"
        nodesep=0.5
        node [shape=record, color=blue width=3.4]
            output_p11 [label="Output P11|2023 Population by dwelling type, NS-SeC\nhh#adults, hh#children, hh#cars,\nage, gender, economic status,\nemployment status, SOC|LSOA"];
            output_p13_4 [label="Output P13.4|2023 Occupancies by dwelling type, NS-SeC\n#adults, #children, #cars|LSOA"];

            pop_targets [label="Population Targets|ONS Population Projections, by age (NTEM), gender|RGN2021"]
            soc_targets [label="SOC Targets|LM&S Projections, by SOC (excluding SOC4), gender| RGN2021"]

        node [style=rounded, color=black]

            output_pop [label="Forecast Population Output|LSOA"];
            output_hhs [label="Forecast Households Output|LSOA"];

        output_p11 -> output_pop;
        pop_targets -> output_pop;
        soc_targets -> output_pop;

        output_p13_4 -> output_hhs;
        output_pop -> output_hhs;

    }