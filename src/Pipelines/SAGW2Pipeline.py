import time
from datetime import datetime
import dateutil.parser
import pandas as pd
import numpy as np

import gc
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale
import scipy.signal as signal

from lib.Log.Log import Logger
from Pipelines.preprocessing import add_elapsed_time, add_first_derivative, split_sequence, evaluate
from Pipelines.ProcessPipeline import ProcessPipeline
from lib.Storage.StorageUtil import StorageUtil
from lib.Cache.CacheUtil import CacheUtil
from lib.Exceptions.CacheExceptions import PipelineUtilNotEnoughItemInCache, PipelineUtilNoSuchPipelineSetting


class SAGW2Pipeline(ProcessPipeline):
    pipeline_name = None
    n_features = 0
    n_steps = 0
    epocs = 0
    smooth_window_size = 0
    prediction_column = ""
    time_stamp_column = ""
    model_metadata = None
    stream = ""
    # training_start_time = ""
    # training_end_time = ""
    result_headers = list()
    training_result_headers = list()
    training_time_delta = None

    def __init__(self, config):
        SAGW2Pipeline.pipeline_name = config.get('pipeline_name')
        SAGW2Pipeline.n_features = config.get('n_features')
        SAGW2Pipeline.n_steps = config.get('n_steps')
        SAGW2Pipeline.p_steps = config.get('p_steps')
        SAGW2Pipeline.epocs = config.get('epocs')
        SAGW2Pipeline.smooth_window_size = config.get('smooth_window_size')

        SAGW2Pipeline.stream = config.get('stream')
        SAGW2Pipeline.prediction_column = config.get('prediction_column')
        SAGW2Pipeline.time_stamp_column = config.get('time_stamp_column')
        SAGW2Pipeline.training_time_delta = self.compute_training_time_delta(config.get('training_data_period'))
        SAGW2Pipeline.model_meta_headers = config.get('model_meta_headers')
        SAGW2Pipeline.training_result_headers = config.get('training_result_headers')

        SAGW2Pipeline.result_headers = config.get('result_headers')

    @staticmethod
    def create_model(config):
        # n_steps = config.get('n_steps')
        n_features = config.get('n_features')

        model = Sequential()

        model.add(Dense(160, activation = "relu", input_dim=n_features, kernel_initializer='random_uniform'))
        model.add(Dense(160, activation = "relu", kernel_initializer='random_uniform' ))
        model.add(Dense(160, activation = "relu" ))
        model.add(Dense(50, activation="relu" ))

        model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
        adam_optimizer = keras.optimizers.Adam(lr=.0005)
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
        Logger.debug('training start at: {}'.format(start_time))

        # create LSTM model
        model = SAGW2Pipeline.create_model(config)

        s_time = config.get('s_time')
        e_time = config.get('e_time')

        if s_time is None or e_time is None:
            s_time, e_time = SAGW2Pipeline.get_training_date_range(config.get('training_time_delta'))

        # get train data
        train_data = SAGW2Pipeline.load_data(type_id, config, s_time, e_time)

        if train_data.empty:
            end_time = time.time()
            err_msg = "No data for {} to {} for training".format(s_time, e_time)
            result_obj = dict(start_time=start_time, end_time=end_time, duration=end_time - start_time, type_id=type_id,
                              pipeline=config.get('pipeline_name'), err=err_msg)
            print(result_obj)
            SAGW2Pipeline.submit_training_result(result_obj, config.get('web_requester'))
            return

        Logger.info('loading data for training: {} * {}'.format(train_data.shape[0], train_data.shape[1]))

        # preprocessing
        train_data["ds"] = add_elapsed_time(train_data, config.get('time_stamp_column'))
        x_train = pd.DataFrame(columns=['workday', 'weekday', 'hour', 'minute'])
        for i in range(train_data.shape[0]):
            t = train_data[config.get('time_stamp_column')][i]
            weekday = t.weekday()

            if weekday<5:
                workday=1
            else:
                workday=0

            hour=t.hour
            minute=t.minute

            x_train = x_train.append({'workday': workday, 'weekday': weekday, 'hour': hour, 'minute': minute}, ignore_index=True)

        # y_train =  pd.Series(signal.medfilt(train_data[prediction_column], config.get('smooth_window_size')))
        train_data["smoothed"] = pd.Series(signal.medfilt(train_data[config.get('prediction_column')], config.get('smooth_window_size')))

        # temporary data cleaning
        train_data["smoothed"].replace(np.inf, 0, inplace=True)
        train_data["smoothed"].replace(-np.inf, -0, inplace=True)
        train_data["smoothed"].replace(np.nan, 0, inplace=True)

        # for i in range(train_data.shape[0]):
        #     if np.isinf(train_data["smoothed"][i]):
        #         pass

        train_min = train_data["smoothed"].min()
        train_max = train_data["smoothed"].max()
        train_scale_factor = train_max - train_min

        train_data["y"] = pd.Series(minmax_scale(train_data["smoothed"]))

        # x, y_train = split_sequence(train_data["y"], config.get('n_steps'))
        # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], config.get('n_features')))

        es = EarlyStopping(monitor = 'val_loss',mode = "min", verbose = 1,patience = 20)
        # mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")

        # cb_list = [es, mc]
        cb_list = [es]
        hst = model.fit(x_train, train_data["y"], epochs=config.get('epocs'), validation_split=0.2, callbacks=cb_list, verbose=0)

        # get training results
        y_train_hat = model.predict(x_train, None)
        # print("prediction is done!")
        train_result = pd.DataFrame()
        # train_result["ds"] = train_data["elapsed_time"][self.n_steps:].reset_index(drop = True)
        train_result["yhat"] = pd.Series(y_train_hat[:, 0])
        train_result["y"] = train_data["y"]
        train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
        train_rmse = evaluate(train_result.yhat, train_result.y)

        data_res = list()
        for i in range(train_result.shape[0]):
            data_res.append(dict(create_date=train_data['ds'][i],
                            y=train_result['y'][i],
                            yhat=train_result['yhat'][i],
                            ydiff=train_result['ydiff'][i]))

        # save processing result
        model_metadata = {'min': train_min, 'max': train_max, 'scale': train_scale_factor,
        "training_start_time": str(s_time),
        "training_end_time": str(e_time),
        "rmse": train_rmse}
        SAGW2Pipeline.save_model(config.get('pipeline_name'), type_id, model_metadata, model, data_res)

        print('model training is done! pipeline: {}, type_id: {}'.format(config.get('pipeline_name'), type_id))

        end_time = time.time()
        result_obj = dict(start_time=start_time, end_time=end_time, duration=end_time - start_time, type_id=type_id,
                          pipeline=config.get('pipeline_name'), model_metadata=model_metadata)

        # return training result by http
        SAGW2Pipeline.submit_training_result(result_obj, config.get('web_requester'))
        return

    @staticmethod
    def predict(type_id, data, config):
        # time.sleep(300)
        # result_obj = dict(type_id=type_id,
        #             pipeline=config.get('pipeline_name'),
        #             create_date=data[config.get('time_stamp_column')],
        #             ep=1,
        #             yhat=1,
        #             threshold=1,
        #             label=False)

        # return result_obj

        cached_data = dict()

        model, meta = SAGW2Pipeline.load_model(type_id, config.get('pipeline_name'))
        train_rmse = meta["rmse"][0]
        threshold = train_rmse

        result_obj = None
        # print('predict model:', model)
        # print('data: ', data)
        # print('predict meta:', meta)

        min_val = meta['min'][0]
        max_val = meta['max'][0]
        scale = meta['scale'][0]

        cur_val = data[config.get('prediction_column')]
        cur_t = data[config.get('time_stamp_column')]
        cached_data[config.get('prediction_column')] = data[config.get('prediction_column')]
        cached_data[config.get('time_stamp_column')] = data[config.get('time_stamp_column')]

        if not isinstance(cur_t, datetime):
            cur_t = dateutil.parser.parse(cur_t)


        weekday = cur_t.weekday()

        if weekday < 5:
            workday = 1
        else:
            workday = 0

        hour = cur_t.hour
        minute = cur_t.minute

        x_test = pd.DataFrame(np.array([[workday, weekday, hour, minute]]), columns=['workday', 'weekday', 'hour', 'minute'])

        predict_results = model.predict(x_test, None)
        yhat = predict_results.item(0)
        yreal = (cur_val - min_val) / scale
        ydiff = abs(yhat - yreal)
        if ydiff > threshold:
            label_hat = 'True'
        else:
            label_hat = 'False'

        cached_data['y'] = yreal
        cached_data['yhat'] = yhat
        cached_data['label_hat'] = label_hat

        # Todo: get data from cache
        try:
            cache = CacheUtil.get_all_records(config.get('pipeline_name'), type_id)
            n_cache = len(cache[config.get('prediction_column')])

            # last_val = cached_data[config.get('prediction_column')][-1]
            # last_t = cached_data[config.get('time_stamp_column')][-1]

            if n_cache < config.get('n_steps'):
                # the cached data is not enough
                pass
            else:
                # if all(cache['label_hat']):
                #     xlabel = True
                # else:
                #     xlabel = False
                labels = list()
                for i in range(len(cache['label_hat'])):
                    if cache['label_hat'][i].decode("utf-8") == 'True':
                        labels.append(True)
                    elif cache['label_hat'][i].decode("utf-8") == 'False':
                        labels.append(False)
                    else:
                        print("cache label error!")

                if all(labels):
                    xlabel = True
                else:
                    xlabel = False

                result_obj = dict(type_id=type_id,
                          pipeline=config.get('pipeline_name'),
                          create_date=cached_data[config.get('time_stamp_column')],
                          ep=cached_data[config.get('prediction_column')],
                          yhat=cached_data['yhat'],
                          threshold=threshold,
                          label=xlabel)

            CacheUtil.set_record(config.get('pipeline_name'), type_id, cached_data)

        except PipelineUtilNotEnoughItemInCache:
            # insert the first record to cache
            # data['x'] = 0
            # data['xhat'] = 0

            CacheUtil.set_record(config.get('pipeline_name'), type_id, cached_data)

        return result_obj
