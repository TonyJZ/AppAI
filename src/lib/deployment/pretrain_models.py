from flask import Flask, redirect, url_for, request
import numpy as np
import ray
import time 
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import _datetime
from dateutil.relativedelta import relativedelta
import dateutil.parser
import requests
from concurrent.futures import ThreadPoolExecutor

from storage.storage_util import *
import applications.demo_alpha_lite.config as appConfig

from algorithm import preprocessing
from algorithm.lstm_model import *


def model_trainer_implementation():
    # Obtain content object
    # content = request.form
    
    for meter_id_item in appConfig.LIST_METER_IDS:

        content = {
            'solution_type': appConfig.TRAINING_SOLUTION_TYPE,
            'meter_id': meter_id_item
        }
        
        solution_type = content['solution_type']
        selected_meters = [content['meter_id']]
        start_time = dateutil.parser.parse('2019-03-01 00:00:00')
        end_time = dateutil.parser.parse('2019-06-01 00:00:00')
        # duration = end_time - start_time
        prediction_column = "ep"

        model_config = appConfig.MODEL_CONFIG

        begin = datetime.datetime.now()
        results = []
        for meter_id in selected_meters:
            print('', meter_id, 'being processed')
            train_data = pd.DataFrame()
            
            date = start_time
            while date.month < end_time.month:
                mon_data = read_raw(meter_id, date)
                train_data = train_data.append(mon_data, ignore_index=True)
                date = date + relativedelta(months=+1)

            train_data.columns = appConfig.RAW_HEADERS
            
            # preprocessing
            train_data["ds"] = preprocessing.add_elapsed_time(train_data)
            train_data["diff"] = preprocessing.add_first_derivative(train_data, prediction_column, "ds")
            train_min = train_data["diff"].min()
            train_max = train_data["diff"].max()
            train_scale_factor = train_max - train_min
            metadata = [] # Hardcoded !!!
            metadata.append([train_min])
            metadata.append([train_max])
            write_model_meta(meter_id, 0, metadata)

            train_data["y"] = pd.Series(minmax_scale(train_data["diff"]))
            
            configs = {}
            configs["n_features"] = model_config["n_features"]
            configs["n_steps"] = model_config["n_steps"]
            configs["epoc"] = model_config["epoc"]
            configs["prediction_column"] = "y"
            configs["model_id"] = meter_id
            
            model = LSTM_Model.remote(configs)
            name, train_result, train_rmse = model.training.remote(train_data, write_model)
            results.append({"name":name, "train_result": train_result,"train_rmse":train_rmse})
            
        for r in results:
            name = ray.get(r["name"])
            train_result = ray.get(r["train_result"])
            train_rmse = ray.get(r["train_rmse"])

            print("save model", name)        
            print('train_result.shape =', train_result.shape)
            print('train_data,shape =', train_data.shape)

            data_res = list()
            for i in range(train_result.shape[0]):
                data_res.append([train_data['create_date'][i+MODEL_CONFIG["n_steps"]], train_result['y'][i], train_result['yhat'][i], train_result['ydiff'][i]])    


            write_training_results(name, solution_type, data_res)

    end = datetime.datetime.now()
    print("total time " + str( end - begin ))

    # return end - begin
    return 'ok'



if __name__ == '__main__':
    ray.init(num_cpus = 4, object_store_memory=100000000)
    model_trainer_implementation()
    