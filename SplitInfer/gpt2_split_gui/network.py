import socket
import torch
import io
import pickle

def receive_data(conn):
    data_len_bytes = conn.recv(4)
    length = int.from_bytes(data_len_bytes, 'big')
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            break
        data += packet
    recv_data = pickle.loads(data)
    return recv_data

def send_data(conn, data_to_send):
    serialized_data = pickle.dumps(data_to_send)
    data_length = len(serialized_data).to_bytes(4, 'big')
    conn.send(data_length)
    conn.send(serialized_data)

