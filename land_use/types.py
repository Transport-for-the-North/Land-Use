# -*- coding: utf-8 -*-
"""
Created on: Tues March 2 14:27:23 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Custom types used in Normits Land Use
"""
import pathlib

from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np

PathLike = Union[str, pathlib.Path]

SegmentAggregationDict = Dict[str, Dict[str, List[Any]]]
FactorsDict = Dict[str, Dict[str, np.array]]
