Model Structure
###############

Introduction
============
This page can be used as a reference for the general flow of data within each of the base models.

Base Year Employment
====================

**Work in progress**

Where the year for geographical area is not stated then it is 2021. Geographies cover England, Scotland and Wales.
SIC Section is the first level (at 1 digit)
SIC Division is the second level (at 2 digit)
SIC Groups is the third level (at 3 digits) which is not used in this process
SIC Class is the forth level (at 4 digit)

SOC has 4 categories, but some of the datasets do not include the full range as SOC=4 represents unemployed people.
For datasets with a SOC Segmentation but where the range is not stated then it will be the full range (1-4).

.. graphviz::

    digraph G {
        rankdir="LR"
        nodesep=0.5
        node [shape=record, color=blue width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                table_1 [label="BRES 2022 Employment LAD|Jobs by LAD, SIC Class|LAD"];
                table_2 [label="BRES 2022 Employment MSOA|Jobs by MSOA, SIC Division|MSOA 2011"];
                table_2a [label="BRES 2022 Employment MSOA SIC splits|SIC Section and SIC Division splits|MSOA 2011"];
                table_3 [label="BRES 2022 Employment LSOA|Jobs by LSOA, SIC Section|LSOA 2011"];
            }
        
        node [shape=record, color=blue width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                table_2a [label="Balanced BRES 2022 Employment MSOA|SIC Division|MSOA 2011"];
                table_3a [label="Balanced BRES 2022 Employment LSOA|SIC Section|LSOA 2011"];
            }
            
        node [shape=record, color=blue width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                table_6 [label="ONS Jobs by SIC and SOC|SIC Section and SOC (1-3)|GOR"];            
            }
        
        node [style=rounded, color=black]
                subgraph cluster_inputs{
                peripheries=0
                rank="same"
                output_e1 [label="Output E1|Jobs by LAD, SIC Class|LAD"];
                output_e2 [label="Output E2|Jobs by MSOA, SIC Division|MSOA"];
                output_e3 [label="Output E3|Jobs by LSOA, SIC Section|LSOA"];
            }
                
        node [shape=record, color=blue width=3.4]
            table_7 [label="Jobs by LSOA|SIC Section and SOC (1-3)|LSOA"];
            table_7a [label="Jobs by LSOA|SIC (Section and Division) and SOC (1-3)|LSOA"];
            table_8 [label="Jobs SIC splits|SIC (Section and Division)|MSOA"];
            table_11 [label="Jobs by LSOA|SIC (Section and Division) and SOC|LSOA"];
            table_10 [label="SOC 4 Factors|SOC 4 proportions by region|GOR"];
        
        node [shape=record, color=blue width=3.4]
            table_4 [label="WFJ 2023|Total workforce jobs by region|GOR"];
            lsoa_job_splits [label="Jobs Splits|Proportions by LSOA\nwithin LAD, allocted by SIC|LSOA"];
            
        node [style=rounded, color=black]
            output_e4 [label="Output E4|Jobs by LSOA, \nSIC (Section and Division), SOC|LSOA"];
            output_e4_2 [label="Output E4.2|Jobs by LSOA, SIC Division,\nSOC weighted to WFJ|LSOA"];
            output_e4_3 [label="Output E4.3|Jobs by LSOA, \nSIC (Section and Division), SOC\ndistribution corrected|LSOA"];
            output_e5 [label="Output E5|Jobs by LSOA, SOC\nSIC (Class, Section, Division)|LSOA"];
            output_e6 [label="Output E6|Jobs by LSOA, SOC\nSIC (Class, Section, Division)\ndistribution corrected|LSOA"];
        

        table_1 -> output_e1;
        output_e1 -> output_e5
        table_1 -> table_2a;
        table_1 -> table_3a;
        table_2 -> table_2a;
        table_2a -> output_e2;
        table_3 -> table_3a;
        table_3a -> output_e3;
        output_e3 -> table_7;
        table_6 -> table_7;
        output_e2 -> table_8
        table_8 -> table_7a
        table_11 -> output_e4
        table_7 -> table_7a
        table_7a -> table_11
        table_10 -> table_11
        table_4 -> output_e4_2
        output_e4 -> output_e4_2
        lsoa_job_splits -> output_e4_3
        output_e4 -> output_e4_3
        output_e4 -> output_e5
        lsoa_job_splits -> output_e6
        output_e5 -> output_e6
    }

Forecast Employment
===================

**Work in progress**

.. graphviz::

    digraph G {
        rankdir="LR"
        nodesep=0.5
        node [shape=record, color=blue width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                table_1 [label="BRES 2022 Employment LAD|Jobs by LAD, SIC Class|LAD"];
                table_2 [label="BRES 2022 Employment MSOA|Jobs by MSOA, SIC Division|MSOA 2011"];
                table_2a [label="BRES 2022 Employment MSOA SIC splits|SIC Section and SIC Division splits|MSOA 2011"];
                table_3 [label="BRES 2022 Employment LSOA|Jobs by LSOA, SIC Section|LSOA 2011"];
            }

        node [shape=record, color=blue width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                table_2a [label="Balanced BRES 2022 Employment MSOA|SIC Division|MSOA 2011"];
                table_3a [label="Balanced BRES 2022 Employment LSOA|SIC Section|LSOA 2011"];
            }

        node [shape=record, color=blue width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                table_6 [label="ONS Jobs by SIC and SOC|SIC Section and SOC (1-3)|GOR"];
            }

        node [style=rounded, color=black]
                subgraph cluster_inputs{
                peripheries=0
                rank="same"
                output_e1 [label="Output E1|Jobs by LAD, SIC Class|LAD"];
                output_e2 [label="Output E2|Jobs by MSOA, SIC Division|MSOA"];
                output_e3 [label="Output E3|Jobs by LSOA, SIC Section|LSOA"];
            }

        node [shape=record, color=blue width=3.4]
            table_7 [label="Jobs by LSOA|SIC Section and SOC (1-3)|LSOA"];
            table_7a [label="Jobs by LSOA|SIC (Section and Division) and SOC (1-3)|LSOA"];
            table_8 [label="Jobs SIC splits|SIC (Section and Division)|MSOA"];
            table_11 [label="Jobs by LSOA|SIC (Section and Division) and SOC|LSOA"];
            table_10 [label="SOC 4 Factors|SOC 4 proportions by region|GOR"];

        node [shape=record, color=blue width=3.4]
            table_4 [label="WFJ 2023|Total workforce jobs by region|GOR"];
            lsoa_job_splits [label="Jobs Splits|Proportions by LSOA\nwithin LAD, allocted by SIC|LSOA"];

        node [style=rounded, color=black]
            output_e4 [label="Output E4|Jobs by LSOA, \nSIC (Section and Division), SOC|LSOA"];
            output_e4_2 [label="Output E4.2|Jobs by LSOA, SIC Division,\nSOC weighted to WFJ|LSOA"];
            output_e4_3 [label="Output E4.3|Jobs by LSOA, \nSIC (Section and Division), SOC\ndistribution corrected|LSOA"];
            output_e5 [label="Output E5|Jobs by LSOA, SOC\nSIC (Class, Section, Division)|LSOA"];
            output_e6 [label="Output E6|Jobs by LSOA, SOC\nSIC (Class, Section, Division)\ndistribution corrected|LSOA"];


        table_1 -> output_e1;
        output_e1 -> output_e5
        table_1 -> table_2a;
        table_1 -> table_3a;
        table_2 -> table_2a;
        table_2a -> output_e2;
        table_3 -> table_3a;
        table_3a -> output_e3;
        output_e3 -> table_7;
        table_6 -> table_7;
        output_e2 -> table_8
        table_8 -> table_7a
        table_11 -> output_e4
        table_7 -> table_7a
        table_7a -> table_11
        table_10 -> table_11
        table_4 -> output_e4_2
        output_e4 -> output_e4_2
        lsoa_job_splits -> output_e4_3
        output_e4 -> output_e4_3
        output_e4 -> output_e5
        lsoa_job_splits -> output_e6
        output_e5 -> output_e6
    }