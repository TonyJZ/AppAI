#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:15:56 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import keras
from appai_lib import my_io
from appai_lib import preprocessing
sns.set(style="dark")

in_file = "../data/processed/april_may_elapsed_filtered_36.csv"
all_data =my_io.load_data(in_file)
counts ={}
grouped = all_data.groupby("meter_id")

for name, group in grouped:
    group = group.sort_values(by=["create_date"])
    group["match"] = (group.elapsed_time - group.elapsed_time.shift() == 300)
    print("false " + str( group.match.sum() - len(group.match)))
    counts[name]=  group.match.sum() - len(group.match) + 1
