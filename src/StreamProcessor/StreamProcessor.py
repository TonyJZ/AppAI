from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import dateutil.parser
from flask import Flask, request, Response
import numpy as np
import ray
import requests
import time 

from lib.Log.Log import Logger
from lib.Common.Auxiliary import Auxiliary
from lib.Config.ConfigUtil import ConfigUtil
from lib.Storage.StorageUtil import StorageUtil
from Pipelines.PipelineFactory import PipelineFactory 


class StreamProcessor:

    stream_to_pipelines_map = {}

    app = Flask(__name__)


    # Main end point for RESTful API
    @app.route('/', methods = ['POST']) 
    def stream_processor_implementation(config_info=None):
        
        # Start timestamp
        dt1 = datetime.now()

        # Obtain json object
        request_json_body = request.get_json()
        Logger.debug('Received request {}'.format(request_json_body))

        resp = Response()

        # Get stream
        stream = None
        try:
            stream = request_json_body.get('stream')
            if stream is None:
                raise Exception
        except:
            Logger.error('Missing stream in content: {}.'.format(request_json_body))
            resp.status_code = 500
            return resp

        # Get proper type_id
        type_id = None
        try:
            type_id = request_json_body.get(stream + '_id')
            if type_id is None:
                raise Exception
        except:
            Logger.error('Missing type_id of stream = {}.'.format(stream))
            resp.status_code = 500
            return resp
        
        # Log
        Logger.debug('Method stream_processor_implementation was called for stream = {} and type_id = {} at {}.'.format(stream, type_id, dt1))

        # Activate all pipelines for that particular stream
        try:
            for pipeline in StreamProcessor.stream_to_pipelines_map.get(stream):

                # result = pipeline.get_class().predict(type_id, request_json_body, pipeline.get_config())
                prediction_remote_method = ray.remote(pipeline.get_class().predict)
                result = ray.get(prediction_remote_method.remote(type_id, request_json_body, pipeline.get_config()))
                Logger.debug('Prediction for stream = %s and type_id (%s_id) = %d resulted in %s.' % (stream, stream, type_id, str(result)))
                resp.set_data('ok')
                if result is None:
                    pass
                else:                   
                    StorageUtil.write_result(type_id, dateutil.parser.parse(request_json_body[pipeline.get_config().get('time_stamp_column')]), pipeline.get_config().get('pipeline_name'), result)

        except Exception as exception:
            Logger.error('Prediction failed for stream = %s and type_id (%s_id) = %d. Exception: %s.' % (stream, stream, type_id, repr(exception)))
            resp.status_code = 500
            # return repr(exception)

        # Processing time
        dt2 = datetime.now()
        dt = dt2 - dt1
        Logger.debug('dt(stream_processor_implementation) = %d us.', dt.microseconds)

        return resp


    def __init__(self):

        # Log
        Logger.info('Stream Processor service initialized.')

        # Load config
        sp_config = ConfigUtil.get_value_by_key('services', 'StreamProcessor')
        streams = ConfigUtil.get_value_by_key('streams')
        
        # Create map of streams - pipelines
        pipeline_factory = PipelineFactory()

        for stream in streams.values():
            pipelines = []
            for pipeline in stream.get('pipelines'):
                pipeline_instance = pipeline_factory.get_pipeline(pipeline.get('pipeline_name'))
                pipelines.append(pipeline_instance)
            StreamProcessor.stream_to_pipelines_map.setdefault(stream.get('stream'), pipelines)

        # print(StreamProcessor.stream_to_pipelines_map)

        ray.init(num_cpus = int(sp_config.get('ray_cpus')), object_store_memory=int(sp_config.get('ray_memory')))
        Logger.info('Ray was initialized.')
        
        Logger.info('Initializing Flask app...')
        StreamProcessor.app.run(debug=False, threaded=True, host="0.0.0.0", port=int(sp_config.get('port')))

        Logger.info('Stream Processor service terminated.')


if __name__ == '__main__':

    sp = StreamProcessor()

