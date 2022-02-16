import time
from datetime import datetime

import dateutil.parser
import numpy as np
import pandas as pd
import scipy.signal as signal
import tensorflow.keras as keras
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

from Pipelines.ProcessPipeline import ProcessPipeline
from Pipelines.preprocessing import add_elapsed_time, add_first_derivative, split_sequence, evaluate
from lib.Cache.CacheUtil import CacheUtil
from lib.Exceptions.CacheExceptions import PipelineUtilNotEnoughItemInCache
from lib.Storage.StorageUtil import StorageUtil


class SAGW1Pipeline(ProcessPipeline):
    pipeline_name = None
    n_features = 0
    n_steps = 0
    epocs = 0
    smooth_window_size = 0
    prediction_column = ""
    time_stamp_column = ""
    model_metadata = None
    stream = ""
    training_time_delta = None
    result_headers = list()
    training_result_headers = list()

    def __init__(self, config):
        SAGW1Pipeline.pipeline_name = config.get('pipeline_name')
        SAGW1Pipeline.n_features = config.get('n_features')
        SAGW1Pipeline.n_steps = config.get('n_steps')
        # SAGW1Pipeline.p_steps = config.get('p_steps')
        SAGW1Pipeline.epocs = config.get('epocs')
        SAGW1Pipeline.smooth_window_size = config.get('smooth_window_size')

        SAGW1Pipeline.stream = config.get('stream')
        SAGW1Pipeline.prediction_column = config.get('prediction_column')
        SAGW1Pipeline.time_stamp_column = config.get('time_stamp_column')
        SAGW1Pipeline.training_time_delta = self.compute_training_time_delta(config.get('training_data_period'))
        SAGW1Pipeline.model_meta_headers = config.get('model_meta_headers')
        SAGW1Pipeline.training_result_headers = config.get('training_result_headers')

        # initialize result headers
        SAGW1Pipeline.result_headers = config.get('result_headers')

        # print(CNAF1Pipeline.result_headers)
        # update results header in StorageUtil
        # StorageUtil.set_pipeline_headers(config.get('pipeline_name'), CNAF1Pipeline.result_headers, None, None)

    @staticmethod
    def create_model(config):
        n_steps = config.get('n_steps')
        n_features = config.get('n_features')

        model = Sequential()
        model.add(LSTM(20, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))
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

        # create LSTM model
        model = SAGW1Pipeline.create_model(config)

        s_time = config.get('s_time')
        e_time = config.get('e_time')

        if s_time is None or e_time is None:
            s_time, e_time = SAGW1Pipeline.get_training_date_range(config.get('training_time_delta'))

        # get train data
        train_data = SAGW1Pipeline.load_data(type_id, config, s_time, e_time)

        if train_data.empty:
            end_time = time.time()
            err_msg = "No data for {} to {} for training".format(s_time, e_time)
            result_obj = dict(start_time=start_time, end_time=end_time, duration=end_time - start_time, type_id=type_id,
                              pipeline=config.get('pipeline_name'), err=err_msg)
            print(result_obj)
            SAGW1Pipeline.submit_training_result(result_obj, config.get('web_requester'))
            return

        # preprocessing
        train_data["ds"] = add_elapsed_time(train_data, config.get('time_stamp_column'))
        train_data["diff"] = add_first_derivative(train_data, config.get('prediction_column'), "ds")

        train_data["smoothed"] = pd.Series(signal.medfilt(train_data["diff"], config.get('smooth_window_size')))

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

        x_train, y_train = split_sequence(train_data["y"], config.get('n_steps'))
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], config.get('n_features')))

        es = EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=20, min_delta=0.005)
        # mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")

        # cb_list = [es, mc]
        cb_list = [es]
        hst = model.fit(x_train, y_train, epochs=config.get('epocs'), validation_split=0.2, callbacks=cb_list,
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
                          "training_start_time": str(s_time), "training_end_time": str(e_time),
                          "rmse": train_rmse}
        SAGW1Pipeline.save_model(config.get('pipeline_name'), type_id, model_metadata, model, data_res)

        print('model training is done! pipeline: {}, type_id: {}'.format(config.get('pipeline_name'), type_id))

        end_time = time.time()
        result_obj = dict(start_time=start_time, end_time=end_time, duration=end_time - start_time, type_id=type_id,
                          pipeline=config.get('pipeline_name'), model_metadata=model_metadata)

        # return training result by http
        SAGW1Pipeline.submit_training_result(result_obj, config.get('web_requester'))
        return

    @staticmethod
    def predict(type_id, data, config):
        # time.sleep(300)
        # result_obj = dict(type_id=type_id,
        #             pipeline=config.get('pipeline_name'),
        #             create_date=data[config.get('time_stamp_column')],
        #             ep=1,
        #             x=1,
        #             xhat=1,
        #             threshold=1,
        #             label=False)
        # return result_obj                        


        cached_data = dict()
        cached_data[config.get('time_stamp_column')] = data[config.get('time_stamp_column')]
        cached_data[config.get('prediction_column')] = data[config.get('prediction_column')]
        # load model and meta
        model, meta = SAGW1Pipeline.load_model(type_id, config.get('pipeline_name'))
        train_rmse = meta["rmse"][0]
        train_th = train_rmse * 3
        min_th = train_th / 3
        max_th = train_th

        result_obj = None
        # print('predict model:', model)
        # print('data: ', data)
        # print('predict meta:', meta)

        min_val = meta['min'][0]
        max_val = meta['max'][0]
        scale = meta['scale'][0]

        # Todo: get data from cache
        try:
            cache = CacheUtil.get_all_records(config.get('pipeline_name'), type_id)
            n_cache = len(cache[config.get('prediction_column')])

            last_val = cache[config.get('prediction_column')][-1]
            last_t = cache[config.get('time_stamp_column')][-1]

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

            cached_data['x'] = (cur_x - min_val) / scale

            if n_cache < config.get('n_steps'):
                # the cached data is not enough
                cached_data['xhat'] = 0
                pass
            else:
                x_test = np.array(cache['x'])
                x_test_hat = np.array(cache['xhat'])

                win_diff = x_test_hat.reshape((config.get('n_steps'), 1)) - x_test.reshape((config.get('n_steps'), 1))
                rolling_thresh = np.std(win_diff) * 3
                thresh = rolling_thresh

                x_test = x_test.reshape((1, config.get('n_steps'), config.get('n_features')))

                predict_results = model.predict(x_test, None)
                xhat = predict_results.item(0)
                xreal = cached_data['x']

                if thresh > max_th:
                    thresh = max_th
                if thresh < min_th:
                    thresh = min_th

                rolling_std = thresh

                xdiff = xhat - xreal
                signRst = np.sign(xdiff)
                xdiff = abs(xdiff)
                if (xdiff > thresh):
                    xhat = xhat - signRst * thresh * 0.8  # softly dragging to real data
                    xlabel = True
                else:
                    xlabel = False

                cached_data['xhat'] = xhat

                result_obj = dict(type_id=type_id,
                                  pipeline=config.get('pipeline_name'),
                                  create_date=cached_data[config.get('time_stamp_column')],
                                  ep=cached_data[config.get('prediction_column')],
                                  x=cached_data['x'],
                                  xhat=cached_data['xhat'],
                                  threshold=rolling_std,
                                  label=xlabel)

            CacheUtil.set_record(config.get('pipeline_name'), type_id, cached_data)

        except PipelineUtilNotEnoughItemInCache:
            # insert the first record to cache
            cached_data['x'] = 0
            cached_data['xhat'] = 0

            CacheUtil.set_record(config.get('pipeline_name'), type_id, cached_data)

        return result_obj
        
        
