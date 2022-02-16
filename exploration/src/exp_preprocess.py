from appai_lib import preprocessing
from appai_lib import my_io
import pandas as pd
from datetime import date
import os


all_nan = [59, 100, 158, 170, 224, 336, 465, 505, 588]
zero_sum =  [6, 7, 8, 9, 10, 16, 18, 23, 30, 36, 39, 41, 49, 55, 59, 70, 84, 97, 98, 100, 110, 112, 119, 131, 137, 158, 170, 180, 186, 192, 196, 208, 212, 213, 216, 224, 229, 230, 233, 235, 236, 237, 244, 246, 252, 256, 259, 260, 264, 266, 269, 270, 275, 279, 280, 282, 284, 288, 294, 298, 300, 309, 310, 315, 316, 323, 325, 328, 332, 334, 335, 336, 338, 340, 345, 351, 352, 356, 359, 364, 376, 382, 392, 398, 406, 410, 415, 420, 422, 423, 426, 429, 433, 437, 441, 442, 451, 455, 456, 457, 461, 462, 464, 465, 471, 477, 479, 480, 482, 486, 496, 497, 499, 505, 513, 523, 526, 535, 538, 544, 547, 555, 559, 573, 578, 585, 586, 588, 590, 596, 599]
selected_meters = [1]#, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
filter_out_meters = all_nan + zero_sum
#holidays = [date(2019,4,5), date(2019,4,6), date(2019,4,7), date(2019,4,13), date(2019,4,14), date(2019,4,20), date(2019,4,21), date(2019,5,1), date(2019,5,2), date(2019,5,3), date(2019,5,4), date(2019,5,11),date(2019,5,12), date(2019,5,18),date(2019,5,19),date(2019,5,25),date(2019,5,26) ]
holidays = [date(2019,4,5), date(2019,4,6), date(2019,4,7), date(2019,4,13), date(2019,4,14), date(2019,4,20), date(2019,4,21),  date(2019,5,11),date(2019,5,12), date(2019,5,18),date(2019,5,19),date(2019,5,25),date(2019,5,26) ]

in_dir = "../data/raw/june/"
out_dir ="../data/processed/june-false-holidayss/"
usecols = ["meter_data_id","meter_id","ep","ap","create_date"]
data_list = []
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
list_files = os.listdir(in_dir)
list_files.sort(key = lambda x: int(x.split(".")[0]))
for file in list_files:
    print("reading " + file)
    data = my_io.load_data(in_dir + file, usecols)
    data_list.append(data)

processed_data = pd.concat(data_list,ignore_index=True)
processed_data = preprocessing.filter_out(processed_data,filter_out_meters)
grouped = processed_data.groupby("meter_id")

for name, group in grouped:
    print("Processing meter " + str(name))
    group = group.sort_values(by=["create_date"])
    group = group.reset_index(drop = True)
    group = preprocessing.add_elapsed_time(group)
    group = preprocessing.add_first_derivative(group,"ep")
    group = preprocessing.add_first_derivative(group,"ep_diff")
    #group = preprocessing.add_scale(group,"ep_diff")
    #group = preprocessing.add_scale(group,"ap")
    group = preprocessing.add_time_features(group, holidays)
    
    group = preprocessing.median_smoothing(group,"ep_diff",5)
    group = preprocessing.median_smoothing(group,"ap",5)
    
    group = preprocessing.moving_average_smoothing(group,"ep_diff",5)
    group = preprocessing.moving_average_smoothing(group,"ap",5)
    
    group = preprocessing.median_smoothing(group,"ep_diff",13)
    group.to_csv(out_dir+str(name) + ".csv",index = False)
