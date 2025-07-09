Data Sources
############

Base Year Model Data
********************

The data sources used in the base employment model are primarily from three sources:

- Business Register and Employment Survey (BRES),
- the Office of National Statistics, and
- the Workforce Jobs (WfJ) Survey.

The sections below describe each input dataset with the:

- unit of the data,
- geographical level the data are provided in,
- the characteristics (or segmentations) that the data include,
- the source location,
- the file location stored on TfN's local drive, and
- the access requirements for the data.

2022 BRES Data
==================================

LAD Employment
--------------

.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Jobs
   * - Geography
     - LAD 2021
   * - Segmentation
     - SIC Class (4 digit)
   * - Source
     - https://www.nomisweb.co.uk/query/construct/summary.asp?mode=construct&version=0&dataset=189
   * - File Location
     - I:/NorMITs Land Use/2023/import/BRES2022/Employment/bres_employment22_lad_4digit_sic.csv
   * - Access
     - Freely available to download

LSOA Employment
---------------

.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Jobs
   * - Geography
     - LSOA 2011
   * - Segmentation
     - SIC Section (1 digit)
   * - Source
     - https://www.nomisweb.co.uk/query/construct/summary.asp?mode=construct&version=0&dataset=189
   * - File Location
     - I:/NorMITs Land Use/2023/import/BRES2022/Employment/bres_employment22_lsoa2011_1digit_sic.csv
   * - Access
     - Freely available to download

MSOA Employment
---------------

.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Jobs
   * - Geography
     - MSOA 2011
   * - Segmentation
     - SIC Division (2 digit)
   * - Source
     - https://www.nomisweb.co.uk/query/construct/summary.asp?mode=construct&version=0&dataset=189
   * - File Location
     - I:/NorMITs Land Use/2023/import/BRES2022/Employment/bres_employment22_msoa2011_2digit_sic_jobs.csv
   * - Access
     - Freely available to download

Office for National Statistics
==============================

ONS Industry (SIC) to Occupation (SOC)
--------------------------------------

.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Jobs
   * - Geography
     - GOR 2021
   * - Segmentation
     - SIC Section (1 digit), SOC group
   * - Source
     - `Office for National Statistics <mailto:Census.CustomerServices@ons.gov.uk>`_
   * - File Location
     - I:/NorMITs Land Use/2023/import/ONS/industry_occupation/population_region_1sic_soc.csv
   * - Access
     - Freely available to download

WFJ
===

WFJ 2023
--------

.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Jobs (Total workforce jobs)
   * - Geography
     - GOR
   * - Segmentation
     - Total
   * - Source
     - `Office for National Statistics <mailto:Census.CustomerServices@ons.gov.uk>`_
   * - File Location
     - I:/NorMITs Land Use/2023/import/BRES2022/Employment/Employment Investigation/WFJ.csv
   * - Access
     - Freely available to download

SOC 4 factors
-------------

.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Percentages (of all residents that are unemployed)
   * - Geography
     - GOR
   * - Segmentation
     - Total
   * - Source
     - TfN internal analysis based on other sources
   * - File Location
     - I:/NorMITs Land Use/2023/import/SOC/Table 8 WFJ-adjusted Land Use SOC4.csv
   * - Access
     - TfN internal analysis


Forecast Model Data
********************

The data sources used in the forecast employment model are from two sources the Labour Market and Skills analysis.

The sections below describe each input dataset with the:

- unit of the data,
- geographical level the data are provided in,
- the characteristics (or segmentations) that the data include,
- the range of years the data include,
- the source location,
- the file location stored on TfN's local drive, and
- the access requirements for the data.

Labour Market and Skills Projections
================================================================================
England Regions, Wales, Scotland LM&S Projections (published March 2023)
------------------------------------------------------------------------
.. list-table::
   :header-rows: 0
   :widths: 1 2
   :stub-columns: 1

   * - Unit
     - Jobs
   * - Geography
     - GOR
   * - Segmentation
     - SIC 1 digit, SOC
   * - Years
     - 2025 to 2035
   * - Source
     - `Labour Market and Skills <https://www.gov.uk/government/publications/labour-market-and-skills-projections-2020-to-2035>`_
   * - Files Location
     - I:/NorMITs Land Use/2023/import/Labour Market and Skills
   * - Access
     - Freely available to download
