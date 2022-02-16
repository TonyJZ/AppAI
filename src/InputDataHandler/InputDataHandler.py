from datetime import datetime
import json
import multiprocessing
import queue
import pika
import requests
import signal
import sys
from threading import Thread
import time

from lib.Common.Auxiliary import Auxiliary
from lib.Config.ConfigUtil import ConfigUtil
from lib.Log.Log import Logger
from lib.Storage.StorageUtil import StorageUtil

import asyncio
import aiohttp
from lib.Web.AsyncWebRequester import AsyncWebRequester

class InputDataHandlerAsyncWebRequester(AsyncWebRequester):
    
    def __init__(self, url):
        super().__init__(url)

    async def post_to_stream_processor(self, content):
        async with aiohttp.ClientSession() as session:
            try:
                Logger.debug('User Request activated training event: %s', content)
                result = await super().post_request(session, content)
                Logger.debug(result)
                return result

            except Exception as e:
                Logger.error('Exception happened while waiting for response. %s', e)
                raise e




class InputDataHandler:

    # HTTP Request (RESTful API) to Stream Processor
    STREAM_PROCESSOR_URL = ''

    NUMBER_PROCESSES = 50

    streams = {}
    routing_key_stream_position = 0

    DATE_STRING_FORMAT = '%Y-%m-%d %H:%M:%S'

    MAX_ATTEMPTS_SEND_REQUEST = 3


    # Threaded function to send post request
    def send_request(self, url, content):

        attempt = 1
        while attempt <= self.MAX_ATTEMPTS_SEND_REQUEST:

            try:
                dt1_send = datetime.now()
                r = requests.post(url, json=content)
                dt2_send = datetime.now()
                dt_send = dt2_send - dt1_send
                Logger.debug('dt(send_request) = %d us.', dt_send.microseconds)
                if r.status_code == 200:
                    Logger.debug('HTTP request has been sent to %s.' % url)
                    break
                Logger.warning('HTTP request has not been sent to %s at attempt = %d/%d. Content: %s.' % (url, attempt, self.MAX_ATTEMPTS_SEND_REQUEST, content))
            except Exception as e:
                Logger.warning('HTTP request has not been sent to %s at attempt = %d/%d. Exception: %s. Content: %s.' % (url, attempt, self.MAX_ATTEMPTS_SEND_REQUEST, e, content))
            finally:
                time.sleep(1)
                attempt += 1

    # Process and submit HTTP request function
    def process_submit_request(self, process_name, task_queue, event=None):

        # Receive task from queue with content
        while True:

            dt1_submit = datetime.now()

            try:
                task = task_queue.get(block=True, timeout=1)
                
                rx_routing_key_list = task.get('routing_key').split('.')
                rx_stream = rx_routing_key_list[task.get('routing_key_stream_position')]
                type_id = int(rx_routing_key_list[-1])

                content = None
                try:
                    content = json.loads(task.get('body').decode('utf-8'))
                    content.setdefault('stream', rx_stream)
                    Logger.debug('JSON content was loaded into an object.')
                except Exception as e:
                    Logger.error('JSON content could not be loaded into an object. Exception: %s.' % e)

                date = datetime.strptime(content.get('create_date'), self.DATE_STRING_FORMAT)

                # Write raw data to storage
                try: 
                    StorageUtil.write_raw(rx_stream, type_id, date, content)
                except Exception as e:
                    Logger.error('HDFS storage error. Exception: %s.' % e)
                    print('HDFS storage error. Exception: %s.' % e)

                # print ('Exception: %s.' % e)

                # Rewrite None values as "None" strings
                for key, value in content.items():
                    if (value == None):
                        content[key] = 'None'
                        
                Logger.debug('%s: received process_submit_request' % (process_name))
                
                dt1_request = datetime.now()
                
                # Send HTTP request
                try:   
                    # Thread(target=self.send_request, args=(self.STREAM_PROCESSOR_URL, content)).start()
                    self.loop.run_until_complete(
                        self.requester.post_to_stream_processor(content))
                    Logger.debug('Thread to send HTTP request has been created.')
                except Exception as e:
                    Logger.warning('Thread to send HTTP request has not been created. Exception: %s.' % e)
                finally:
                    dt2_request = datetime.now()
                    dt_request = dt2_request - dt1_request
                    Logger.debug('dt(process_submit_request) = %d us.' % dt_request.microseconds)

            except queue.Empty:
                pass
                # Logger.warning('%s: process_submit_request was not properly received' % (process_name))

            if event.is_set():
                Logger.info('Event for process %s was caught' % process_name)
                break

            dt2_submit = datetime.now()
            dt_submit = dt2_submit - dt1_submit
            # Logger.debug('dt(process_submit_request) = %d us.' % dt_submit.microseconds)

    # MQ Callback
    def callback(self, ch, method, properties, body):
        dt1_callback = datetime.now()
        
        Logger.debug('Received message from MQ with routing key %s.', method.routing_key)

        # Retrieve stream information
        task = {
            "routing_key": method.routing_key,
            "routing_key_stream_position": self.routing_key_stream_position,
            "body": body
        }

        # Add task to queue
        try:
            self.task_queue.put(task)
        except Exception as e:
            Logger.error('Failed to add task :%s to task queue. Exception: %s.', task, e)

        dt2_callback = datetime.now()
        dt_callback = dt2_callback - dt1_callback
        Logger.debug('dt(callback) = %d us.' % dt_callback.microseconds)

    # Signal handler
    def signal_handler(self, received_signal, frame): 
        if (received_signal == signal.SIGINT):
            Logger.debug('SIGINT received (CTRL+C) with frame %s' % frame) 
            for name_process, item_process in self.processes.items():
                try:
                    item_process.get('event').set()
                    Logger.info('Event for process %s was set.' % name_process)
                except Exception as e:
                    Logger.error('Event for process %s was not set. Exception: %s.' % (name_process, e))
            sys.exit(0)


    # Main function
    def __init__(self):
        
        # Log
        Logger.info('Input Data Handler module initialized.')

        # Set signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        Logger.debug('SIGINT signal handler set.')

        # Load config
        idh_config = ConfigUtil.get_value_by_key('services', 'InputDataHandler')
        stream_processor_ip_port = idh_config.get('stream_processor_ip_port')
        self.STREAM_PROCESSOR_URL = 'http://' + str(stream_processor_ip_port)

        # Initialize pool of processes
        self.processes = {}

        self.requester = InputDataHandlerAsyncWebRequester(self.STREAM_PROCESSOR_URL)
        # setup event loop for main thread
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        self.loop = asyncio.new_event_loop()

        # Initialize processes
        try:
            self.task_queue = multiprocessing.Queue()
            Logger.info('Multiprocessing task queue was created.')
        except Exception as exception_task_queue:
            Logger.critical('Multiprocessing task queue could not be created.')

        for index_process in range(self.NUMBER_PROCESSES):
            process_name = 'P%i' % index_process
            try: 
                event = multiprocessing.Event()
                new_process = multiprocessing.Process (target = self.process_submit_request, args = (process_name, self.task_queue), kwargs = {'event' : event})
                new_process.start()
                self.processes.setdefault(process_name, {'process': new_process, 'event': event})
                Logger.info('Process %s was created.' % process_name)
            except Exception as e:
                Logger.warning('Process %s was not created. Exception: %s.' % (process_name, e))

        # MQ Subscriber (messages from IoT Client Simulator)
        mq = ConfigUtil.get_value_by_key('resources', 'mq')
        credentials = pika.credentials.PlainCredentials(mq.get('mq_host_username'), mq.get('mq_host_password'))
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=mq.get('mq_host_ip'), credentials=credentials))
        Logger.info('MQ connection with %s successfully stablished.' % mq.get('mq_host_ip'))
        channel = connection.channel()

        routing_key = idh_config.get('routing_key')
        routing_key_list = routing_key.split('.')
        self.routing_key_stream_position = routing_key_list.index('stream')

        self.streams = ConfigUtil.get_value_by_key('streams')
        for item in self.streams.values():
            routing_key_list[self.routing_key_stream_position] = item.get('stream')

            stream_routing_key = ''
            queue_name = ''
            for key in routing_key_list:
                stream_routing_key += (key + '.')
                queue_name += (key + '_')
            stream_routing_key = stream_routing_key.rstrip('.')
            queue_name = queue_name.rstrip('_')
            queue_name = queue_name.rstrip('_#')
            
            channel.queue_declare(queue=queue_name)
            channel.queue_bind(exchange='amq.topic', queue=queue_name, routing_key=stream_routing_key)
            channel.basic_consume(queue=queue_name, on_message_callback=self.callback, auto_ack=True)

            print('Routing key added (stream_id = %d): %s' % (item.get('stream_id'), stream_routing_key))
            Logger.info('Routing key added (stream_id = %d): %s' % (item.get('stream_id'), stream_routing_key))

        Logger.debug('The routing keys for %d streams were set.' % len(self.streams))
        
        # Start consuming (subscriber)
        print('\n[*] Waiting for messages. To exit press CTRL+C.\n')
        Logger.info('MQ Callback function set. Ready to start consuming.')
        channel.start_consuming()
        

if __name__ == '__main__':
    idh = InputDataHandler()