# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:35:14 2025

@author: kwill
"""

import numpy as np
import random
import tabula
import matplotlib as plt
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU

# convert pdf file to csv
pdf_path = 'C:/Users/kwill/Downloads/3_57_CAADT.pdf'
csv_path = 'C:/Users/kwill/Desktop/okaloosa.csv'


