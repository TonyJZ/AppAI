#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:07:26 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
from appai_lib import vis
import datetime

exp_name = "dnn_2_false_holiday"
begin = datetime.datetime.now()
vis.vis_result(exp_name,y_lim = False)
print("total time taken " + str(datetime.datetime.now() - begin) )
