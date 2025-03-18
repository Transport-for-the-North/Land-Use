Forecast Model Structure
####################

Introduction
============
This page can be used as a reference for the general flow of data within each of the base models.

Population
==========

**Work in progress**

.. graphviz::


    When forecast year is <= 2043
    digraph G {
        rankdir="LR"

        national_base_2018_forecast
        national_base_2021_forecast

        national_fy_2018_forecast
        national_fy_2021_forecast


        node [shape=record, color=blue width=2]
        adj_base_year [label="adj_base_year|regional_base_2021_forecast * uplift_base_factor"]
        uplift_base_factor [label="uplift_base_factor|national_base_2021_forecasts / national_base_2018_forecasts"]

        adj_fy [label="adj_forecast_year|regional_future_forecast_2021 * uplift_forecast_factor"]
        uplift_fy_factor [label="adj_fy_factor|national_fy_2021_forecasts / national_fy_2018_forecasts"]


        adj_growth_factor [label="adj_growth_factor|adj_future_year / adjusted_base_year"]

        national_base_2018_forecast -> uplift_base_factor
        national_base_2021_forecast -> uplift_base_factor
        regional_base_2021_forecast -> adj_base_year
        uplift_base_factor -> adj_base_year
        adj_base_year -> adj_growth_factor

        national_fy_2018_forecast -> uplift_fy_factor
        national_fy_2021_forecast -> uplift_fy_factor
        regional_fy_2021_forecast -> adj_fy
        uplift_fy_factor -> adj_fy
        adj_fy -> adj_growth_factor

    }

    When forecast year is > 2043, then the regional_fy_2021_forecast is instead (regional_fy_forecast) and
    created as follows
    
    digraph G {
        rankdir="LR"


        national_2043_2018_forecast
        national_fy_2018_forecast

        regional_2043_2021_forecast


        node [shape=record, color=blue width=2]


        national_growth_factor [label="national_growth_factor_2043_to_fy|national_fy_2018_forecast/national_2043_2018_forecast"]
        national_growth_factor [label="national_growth_factor_2043_to_fy|national_fy_2018_forecast/national_2043_2018_forecast"]

        national_2043_2018_forecast -> national_growth_factor
        national_fy_2018_forecast -> national_growth_factor

        regional_2043_2021_forecast -> regional_fy_forecast
        national_growth_factor -> regional_fy_forecast
    }
