import time
from datetime import datetime

import dateutil.parser
import numpy as np
import pandas as pd
from numpy.random import seed
from sklearn.preprocessing import minmax_scale

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras

from lib.Storage.StorageUtil import StorageUtil
from lib.Cache.CacheUtil import CacheUtil
from Pipelines.ProcessPipeline import ProcessPipeline
from Pipelines.preprocessing import add_elapsed_time, add_first_derivative, split_sequence, evaluate
from lib.Exceptions.CacheExceptions import PipelineUtilNotEnoughItemInCache


class CNAF1Pipeline(ProcessPipeline):
    pipeline_name = None
    n_features = 0
    n_steps = 0
    p_steps = 0
    epocs = 0

    prediction_column = ""
    time_stamp_column = ""
    model_meta_headers = None
    stream = ""
    training_time_delta = None

    result_headers = list()
    training_result_headers = list()

    def __init__(self, config):
        CNAF1Pipeline.pipeline_name = config.get('pipeline_name')
        CNAF1Pipeline.n_features = config.get('n_features')
        CNAF1Pipeline.n_steps = config.get('n_steps')
        CNAF1Pipeline.p_steps = config.get('p_steps')
        CNAF1Pipeline.epocs = config.get('epocs')

        CNAF1Pipeline.stream = config.get('stream')
        CNAF1Pipeline.prediction_column = config.get('prediction_column')
        CNAF1Pipeline.time_stamp_column = config.get('time_stamp_column')
        CNAF1Pipeline.training_time_delta = self.compute_training_time_delta(config.get('training_data_period'))
        CNAF1Pipeline.model_meta_headers = config.get('model_meta_headers')
        CNAF1Pipeline.training_result_headers = config.get('training_result_headers')

        # initialize result headers
        CNAF1Pipeline.result_headers = config.get('result_headers')
        for i in range(CNAF1Pipeline.p_steps):
            result_name = 'y{:d}'.format(i)
            CNAF1Pipeline.result_headers.append(result_name)

        # print(CNAF1Pipeline.result_headers)
        # update results header in StorageUtil
        StorageUtil.set_pipeline_headers(config.get('pipeline_name'), CNAF1Pipeline.result_headers, None, None)

    @staticmethod
    def create_model(config):
        n_steps = config.get('n_steps')
        n_features = config.get('n_features')

        model = Sequential()
        model.add(LSTM(120, activation="relu", input_shape=(n_steps, n_features),
                       kernel_initializer='random_uniform', return_sequences=True))
        model.add(LSTM(120, activation="relu", input_shape=(n_steps, n_features),
                       kernel_initializer='random_uniform'))
        model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
        adam_optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam_optimizer, loss='mse')

        return model

    @staticmethod
    def load_model(type_id, pipeline_name):
        return StorageUtil.read_model_and_meta(pipeline_name, type_id)

    @staticmethod
    def save_model(pipeline, type_id, meta, model, train_results):
        model_save_time = datetime.now()
        StorageUtil.write_model_and_meta(pipeline, type_id, model, meta, model_save_time)
        StorageUtil.write_training_results(type_id, pipeline, train_results, model_save_time)

    @staticmethod
    def load_data(type_id, config, s_time, e_time):

        train_data = StorageUtil.read_raw_duration(config.get('stream'), type_id, config.get('time_stamp_column'),
                                                   s_time, e_time)
        return train_data

    @staticmethod
    def train(type_id, config):
        start_time = time.time()

        # create model
        model = CNAF1Pipeline.create_model(config)

        s_time = config.get('s_time')
        e_time = config.get('e_time')

        if s_time is None or e_time is None:
            s_time, e_time = CNAF1Pipeline.get_training_date_range(config.get('training_time_delta'))

        # get train data
        train_data = CNAF1Pipeline.load_data(type_id, config, s_time, e_time)

        if train_data.empty:
            end_time = time.time()
            err_msg = "No data for {} to {} for training".format(s_time, e_time)
            result_obj = dict(start_time=start_time, end_time=end_time, duration=end_time - start_time, type_id=type_id,
                              pipeline=config.get('pipeline_name'), err=err_msg)
            print(result_obj)
            CNAF1Pipeline.submit_training_result(result_obj, config.get('web_requester'))
            return

        # clean the data, as the sales data contain some NaN
        train_data = train_data.fillna(0)

        # preprocessing
        train_data["ds"] = add_elapsed_time(train_data, config.get('time_stamp_column'))
        train_data["diff"] = add_first_derivative(train_data, config.get('prediction_column'), "ds")

        # fix the data issues, will be deleted after testing
        train_data["diff"].replace(np.inf, 0, inplace=True)
        train_data["diff"].replace(-np.inf, -0, inplace=True)
        train_data["diff"].replace(np.nan, 0, inplace=True)

        train_min = train_data["diff"].min()
        train_max = train_data["diff"].max()
        train_scale_factor = train_max - train_min

        train_data["y"] = pd.Series(minmax_scale(train_data["diff"]))

        x_train, y_train = split_sequence(train_data["y"], config.get('n_steps'))
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], config.get('n_features')))

        es = EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=20, min_delta=0.0005)
        # Save the model after every epoch.
        # mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")

        # cb_list = [es, mc]
        cb_list = [es]
        # train the model
        hst = model.fit(x_train, y_train, epochs=config.get('epocs'), validation_split=0.20, callbacks=cb_list,
                        verbose=0)

        # get training results
        y_train_hat = model.predict(x_train, None)
        # print("prediction is done!")
        train_result = pd.DataFrame()
        # train_result["ds"] = train_data["elapsed_time"][self.n_steps:].reset_index(drop = True)
        train_result["yhat"] = pd.Series(y_train_hat[:, 0])
        train_result["y"] = pd.Series(y_train)
        train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
        train_rmse = evaluate(train_result.yhat, train_result.y)

        if train_data.shape[0] - config.get('n_steps') == train_result.shape[0]:
            print('training result size is correct')

        data_res = list()
        for i in range(train_result.shape[0]):
            data_res.append(dict(create_date=train_data['ds'][i + config.get('n_steps')],
                                 y=train_result['y'][i],
                                 yhat=train_result['yhat'][i],
                                 ydiff=train_result['ydiff'][i]))

        # save processing result
        model_metadata = {'min': train_min, 'max': train_max, 'scale': train_scale_factor,
                          "training_start_time": str(s_time),
                          "training_end_time": str(e_time),
                          "rmse": train_rmse}
        CNAF1Pipeline.save_model(config.get('pipeline_name'), type_id, model_metadata, model, data_res)

        print('model training is done! pipeline: {}, type_id: {}'.format(config.get('pipeline_name'), type_id))

        end_time = time.time()
        result_obj = dict(start_time=start_time, end_time=end_time, duration=end_time - start_time, type_id=type_id,
                          pipeline=config.get('pipeline_name'), model_metadata=model_metadata)

        # return training result by http
        CNAF1Pipeline.submit_training_result(result_obj, config.get('web_requester'))
        return

    @staticmethod
    def predict(type_id, data, config):
        
        # time.sleep(300)
        # n_results = config.get('p_steps')
        # results_header = config.get('result_headers')
        # result_obj = dict()
        # for i in range(n_results):
        #     result_obj[results_header[i+1]] = 0
        
        # result_obj[config.get('time_stamp_column')] = data[config.get('time_stamp_column')]
        # result_obj['type_id']=type_id
        # result_obj['pipeline']=config.get('pipeline_name')

        # return result_obj


        # load model and meta
        model, meta = CNAF1Pipeline.load_model(type_id, config.get('pipeline_name'))

        # print('predict model:', model)
        print('data: ', data)
        print('predict meta:', meta)

        min_val = meta['min'][0]
        max_val = meta['max'][0]
        scale = meta['scale'][0]

        # Todo: get data from cache
        try:
            cached_data = CacheUtil.get_all_records(config.get('pipeline_name'), type_id)
            n_cache = len(cached_data[config.get('prediction_column')])

            last_val = cached_data[config.get('prediction_column')][-1]
            last_t = cached_data[config.get('time_stamp_column')][-1]

            cur_val = data[config.get('prediction_column')]
            cur_t = data[config.get('time_stamp_column')]

            if not isinstance(cur_t, datetime):
                cur_t = dateutil.parser.parse(cur_t)
                last_t = dateutil.parser.parse(last_t)

            dt = (cur_t - last_t).total_seconds()
            dv = cur_val - last_val

            try:
                cur_x = dv / dt
            except ZeroDivisionError:
                cur_x = 0

            data['x'] = (cur_x - min_val) / scale

            CacheUtil.set_record(config.get('pipeline_name'), type_id, data)

            if n_cache < config.get('n_steps'):
                # the cached data is not enough
                return None
            else:
                x_test = cached_data['x'][1:]
                x_test.append(data['x'])

                yhats = list()
                predict_results = list()
                temp_ps = list()

                for i in range(0, config.get('p_steps')):
                    x = list(x_test[i:config.get('n_steps')]) + temp_ps[0:i]
                    x = np.array(x)
                    x = x.reshape((1, config.get('n_steps'), config.get('n_features')))
                    yhat = model.predict(x, None)
                    temp_ps.append(yhat[0][0])
                    yhats.append(yhat[0][0])

                    result = yhat[0][0] * scale + min_val + cur_val
                    predict_results.append(result)

        except PipelineUtilNotEnoughItemInCache:
            # insert the first record to cache
            data['x'] = 0
            CacheUtil.set_record(config.get('pipeline_name'), type_id, data)
            return None

        n_results = len(predict_results)
        results_header = config.get('result_headers')
        result_obj = dict()
        for i in range(n_results):
            result_obj[results_header[i+1]] = predict_results[i]
        
        result_obj[config.get('time_stamp_column')] = data[config.get('time_stamp_column')]
        result_obj['type_id']=type_id
        result_obj['pipeline']=config.get('pipeline_name')

        return result_obj
