Parameter Estimation
####################

Model Specification Summary
===========================

The below figure shows a summary of the general process that has been done to
estimate and calibrate NorCOM.

.. graphviz::

    digraph G {
        rankdir="LR"
        nodesep=0.5
        node [shape=record, color=purple width=3.4]
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                nts_data [label="National Travel Survey|2002 to 2023\n(excluding 2020)|Department for Transport" penwidth=3]
            }
            subgraph cluster_inputs{
                peripheries=0
                rank="same"
                income_data_1 [label="Average Household Income (MSOA)|2020|Office of National Statistics" penwidth=3]
                income_data_2 [label="Average Weekly Earnings (GB)|2000-2025|Office of National Statistics" penwidth=3]
            }
        node [shape=record, color=teal width=3.4]
            subgraph cluster_inputs{
                    peripheries=0
                    rank="same"
                zonal_income [label="Average Zonal Income (MSOA)|2021 or 2023|Office of National Statistics" penwidth=3]
                area_type_data [label="TfN Area Type (LSOA)|All years|Transport for the North" penwidth=3]
                _2021_hh [label="Households (LSOA)|2021 or 2023|Land Use Population Model" penwidth=3]
                census [label="Number of Households by\nCar Availability (LSOA)|2021|Census" penwidth=3]
                dvla [label="Number of Registered Vehicles (LSOA)|2023|DVLA" penwidth=3]
            }
        node [style=rounded, color=orange]
            subgraph cluster_inputs{
                    peripheries=0
                    rank="same"
                logit [label="Logit Model Estimation" fontsize=20 penwidth=3];
                calibration [label="Zonal Calibration" fontsize=20 penwidth=3];
            }
        node [style=rounded, color=magenta]
            subgraph cluster_inputs{
                    peripheries=0
                    rank="same"
                norcom [label="NorCOM" penwidth=3 fontsize=20];

            }

        nts_data -> logit [penwidth=3];
        logit -> calibration [penwidth=3];
        census -> calibration [penwidth=3];
        dvla -> calibration [penwidth=3];
        income_data_1 -> zonal_income [penwidth=3];
        income_data_2 -> zonal_income [penwidth=3];
        zonal_income -> calibration [penwidth=3];
        area_type_data -> calibration [penwidth=3];
        _2021_hh -> calibration [penwidth=3];
        calibration -> norcom [penwidth=3];
    }

Estimation Data
===============

The Department for Transport's National Travel Survey (NTS) has been used to estimate
the logit model parameters.

The NTS has detailed records of households, and their attributes, which allows us
to estimate the impact of specific household attributes on the level of household
car ownership.

TfN's `caf.brain <https://github.com/Transport-for-the-North/caf.brain>`_ repository
has been used to estimate these logit models. The exact specification and results of
this estimation can be found here: :code:`I:/NorMITs NorCOM/AECOM working/Model Comparison v38.xlsx`

Estimation Results
==================

A number of logit model forms were tested, but the current logit model parameters
that are used in NorCOM are:

+-------------+-----------------------------------------+----------------+----------------+
| Feature     | Category (if relevant)                  | 0v1+ Parameter | 1v2+ Parameter |
+=============+=========================================+================+================+
| HH Adults   | No adults or 1 adult in household       | 0.00           | -2.59          |
+             +-----------------------------------------+----------------+----------------+
|             | 2 adults in household                   | 0.84           | 0.00           |
+             +-----------------------------------------+----------------+----------------+
|             | 3 or more adults in household           | 0.54           | 0.90           |
+-------------+-----------------------------------------+----------------+----------------+
| HH Children | No children in household                | 0.00           | 0.00           |
+             +-----------------------------------------+----------------+----------------+
|             | At least 1 child in household           | 0.31           | 0.20           |
+-------------+-----------------------------------------+----------------+----------------+
| HH Income   |                                         | 0.49           | 0.63           |
+-------------+-----------------------------------------+----------------+----------------+
| NS-SeC      | HRP managerial / professional           | 0.94           | 0.00           |
+             +-----------------------------------------+----------------+----------------+
|             | HRP intermediate / technical            | 0.62           | 0.14           |
+             +-----------------------------------------+----------------+----------------+
|             | HRP semi-routine / routine              | 0.00           | -0.46          |
+             +-----------------------------------------+----------------+----------------+
|             | HRP never worked / long-term unemployed | -0.79          | -0.79          |
+             +-----------------------------------------+----------------+----------------+
|             | HRP no category (including FT student)  | -0.07          | -0.12          |
+-------------+-----------------------------------------+----------------+----------------+
| House Type  | Detached                                | 0.56           | 0.46           |
+             +-----------------------------------------+----------------+----------------+
|             | Semi-Detached                           | 0.00           | 0.00           |
+             +-----------------------------------------+----------------+----------------+
|             | Terrace                                 | -0.41          | -0.37          |
+             +-----------------------------------------+----------------+----------------+
|             | Flat / Maisonette                       | -1.03          | -0.89          |
+             +-----------------------------------------+----------------+----------------+
|             | Other                                   | -1.07          | -0.51          |
+-------------+-----------------------------------------+----------------+----------------+
| Area Type   | Inner London                            | -1.37          | -1.54          |
+             +-----------------------------------------+----------------+----------------+
|             | Outer London                            | -0.28          | -0.53          |
+             +-----------------------------------------+----------------+----------------+
|             | Major City                              | -0.24          | -0.06          |
+             +-----------------------------------------+----------------+----------------+
|             | City                                    | 0.00           | 0.00           |
+             +-----------------------------------------+----------------+----------------+
|             | Town                                    | 0.29           | 0.29           |
+             +-----------------------------------------+----------------+----------------+
|             | Rural                                   | 0.76           | 0.62           |
+-------------+-----------------------------------------+----------------+----------------+
| Intercept   |                                         | 0.30           | -0.08          |
+-------------+-----------------------------------------+----------------+----------------+

Calibration Adjustments
=======================
