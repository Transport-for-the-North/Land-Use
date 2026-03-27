Population Model
################

Model Specification Summary
===========================

The population model provides an estimate of the number of people and the number
of households at LSOA (2021) zone level for a given year.

As currently specified, the population model requires the following dimensions:

.. list-table::
   :header-rows: 1
   :widths: 1 4 1
   :stub-columns: 1

   * - Dimension
     - Categories
     - Attribute

   * - Sex
     - 2 Categories
        - Male
        - Female
     - Individual

   * - Age
     - 9 Categories
        - Up to 4 years
        - 5-9
        - 10-15
        - 16-19
        - 20-34
        - 35-49
        - 50-64
        - 65-74
        - 75+
     - Individual

   * - Household Composition
     - 7 Categories
        - One person household (age 66+)
        - One person household – other
        - Single family household – lone parent
        - Single family household – couple no children
        - Single family household – couple all children non-dependent
        - Single family household – couple dependent children
        - Other households
     - Household

   * - NS-SeC of Household Representative Person
     - 5 Categories
        - HRP managerial / professional
        - HRP intermediate / technical
        - HRP semi-routine / routine
        - HRP never worked or in long-term unemployment
        - HRP full time student
     - Household

   * - Car Availability
     - 3 Categories
        - No vehicles
        - 1 vehicle
        - 2+ vehicles
     - Household

   * - Economic Status
     - 6 Categories
        - Economically active employees
        - Economically active unemployed
        - Economically active employed students
        - Economically active unemployed students
        - Economically inactive
        - Economically inactive students
     - Individual

   * - Employment Status
     - 5 Categories
        - Full time
        - Part time
        - Unemployed
        - Students
        - Non-working age (i.e. children)
     - Individual

   * - Occupation Type
     - 4 Categories
        - SOC group 1 (Managers, directors, and senior officials, Professional occupations, and Associate professional occupations.)
        - SOC group 2 (Administrative and secretarial occupations, Skilled trades occupations, Caring, leisure, and other service occupations, and Sales and customer service occupations.)
        - SOC group 3 (Process, plant, and machine operatives, and Elementary occupations.)
        - SOC group 4 (Students, Children, and Unemployed.)
     - Individual



More Information
================

.. toctree::
   :maxdepth: 1

   population/data_sources
   population/segmentation
   population/model_structure
   population/assumptions_limitations
