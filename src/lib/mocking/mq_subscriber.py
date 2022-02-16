#!/usr/bin/env python
import pika

credentials = pika.credentials.PlainCredentials('guest', 'guest')

connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.10.10.111', credentials=credentials))
channel = connection.channel()


# channel.queue_declare(queue='input_data_handler')
# channel.queue_bind(exchange='amq.topic', queue='input_data_handler', routing_key='cnaf.store.0.#')
# channel.queue_bind(exchange='amq.topic', queue='input_data_handler', routing_key='sagw.meter.1.#')
# channel.queue_bind(exchange='amq.topic', queue='input_data_handler', routing_key='sagw.meter.2.#')

# def callback(ch, method, properties, body):
#     print(" [x] Received %s: %r" % (method, body))

# channel.basic_consume(queue='input_data_handler', on_message_callback=callback, auto_ack=True)


channel.queue_declare(queue='cnaf-sagw_store')
channel.queue_bind(exchange='amq.topic', queue='cnaf-sagw_store', routing_key='cnaf-sagw.store.#')
channel.queue_declare(queue='cnaf-sagw_meter')
channel.queue_bind(exchange='amq.topic', queue='cnaf-sagw_meter', routing_key='cnaf-sagw.meter.#')

i = 0

def callback(ch, method, properties, body):
    global i
    print(" [x] Received %s: %r" % (method, body))
    # if i % 1000 == 0:
    #     print('%d ' % i)
    # i += 1

channel.basic_consume(queue='cnaf-sagw_store', on_message_callback=callback, auto_ack=True)
channel.basic_consume(queue='cnaf-sagw_meter', on_message_callback=callback, auto_ack=True)


print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()