import time
import zmq

from struct import *

context = zmq.Context()
front = context.socket(zmq.ROUTER)
back = context.socket(zmq.DEALER)
front.bind("tcp://*:5555")
back.bind("tcp://*:5550")

print('Server Running.')

poller = zmq.Poller()
poller.register(front,zmq.POLLIN)
poller.register(back,zmq.POLLIN)

run = True

while run:
    socks = dict(poller.poll())

    if socks.get(front) == zmq.POLLIN:
        message = front.recv_multipart()
        back.send_multipart(message)

    if socks.get(back) == zmq.POLLIN:
        message = back.recv_multipart()
        front.send_multipart(message)

front.close()
back.close()