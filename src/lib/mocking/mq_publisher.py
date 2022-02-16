#!/usr/bin/env python
import datetime
import pika
import time

credentials = pika.credentials.PlainCredentials('guest', 'guest')

# Disable heartbeats at broker (RabbitMQ)
connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.10.10.111', credentials=credentials, heartbeat=0))
channel = connection.channel()

record = '{"create_date": "%s", "store_id": 1, "items": 0, "sales": 1001.0}' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

channel.basic_publish(exchange='amq.topic', routing_key='cnaf-sagw.store.1', body=record)
print('Published: %s' % record)

# time.sleep(10)

# channel.basic_publish(exchange='amq.topic', routing_key='cnaf-sagw.store.1', body=record)
# print('Published: %s' % record)

# time.sleep(200)

# channel.basic_publish(exchange='amq.topic', routing_key='cnaf-sagw.store.1', body=record)
# print('Published: %s' % record)

# for i in range(200):
#     print('', i, ': ', connection.is_open)
#     time.sleep(1)
    
connection.close()

