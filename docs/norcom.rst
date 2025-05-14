NorCOM
######

Model Specification Summary
===========================

NorCOM is TfN's household-based Northern Car Ownership Model. NorCOM predicts the
probability that a given household has 0, 1, or 2 or more cars available, based on
specific household characteristics.

NorCOM is primarily needed to provide car ownership forecasts for future modelling
years and scenarios. It is also needed to “forecast” car ownership in the base year
of the population model (2023), to allow a rebase from the 2021 Census year.

This current version of NorCOM is based on a nested logit model of two models:

- the 0 vs 1+ car model (i.e. the likelihood of a house owning at least one car), and
- the 1 vs 2+ car model (i.e. the likelihood of a house owning more than one car).

These likelihoods are then applied to the household-level population model outputs
to approximate the number of households in each zone that have different levels of
car availability.

More Information
================

.. toctree::
   :maxdepth: 1

   norcom/estimation
   norcom/application
   norcom/assumptions_limitations
