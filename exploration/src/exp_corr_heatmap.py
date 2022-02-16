#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:11:00 2019

@author: Toukir
"""
from appai_lib import preprocessing
from appai_lib import my_io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="dark")

all_data = my_io.load_data("../data/processed/april_may_elapsed_filtered.csv")
#all_data = my_io.load_data("../data/processed/april_may_elapsed_filtered_36.csv")

ap_data = pd.DataFrame()
grouped = all_data.groupby("meter_id")
for name, group in grouped:
    group = group.sort_values(by=["elapsed_time"])
    group = group.reset_index()
    ap = group.ap
    ap_data[name] = ap

corr_data = ap_data.corr()
#with PdfPages('meter_correlation.pdf') as pdf:
print("drawing")
fig = plt.figure(figsize =(250,250))
ax = plt.subplot(1,1,1)
ax.set_title("Meter correlation (ap)")
sns.heatmap(ax = ax, data = corr_data,center = 0.05,annot=True,cmap="RdBu")
plt.xlabel("Meter id")
plt.ylabel("Meter id")
fig.savefig("meter_correlation.png")
    #pdf.savefig()