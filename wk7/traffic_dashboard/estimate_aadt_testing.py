# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:00:50 2025

@author: kwill
"""
import numpy as np

def estimate_aadt_linear(aadt_values):
    x = np.arange(len(aadt_values))
    coef = np.polyfit(x, aadt_values, 1)
    return round(coef[0] * len(aadt_values) + coef[1], 2)

actual_aadts = [150, 200, 150, 200, 150]

aadts = [200, 150, 200]

future_aadt = estimate_aadt_linear(aadts)