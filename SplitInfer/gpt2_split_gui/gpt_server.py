import socket

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from arch.arch import GPT2LMHeadModel
from arch.server import GPT2LMHeadModelServer

from network import *

model = GPT2LMHeadModel.from_pretrained('./gpt2-model')
server = GPT2LMHeadModelServer(model, 2)

tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-model', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

server.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
server.to(device)

def generate(feature, task_id):
    with torch.no_grad():
        feature = feature.to(device)
        outputs = server(feature, task_id)
        next_token_logits = outputs.logits[:, -1, :]

        top_k_values, _ = torch.topk(next_token_logits, 20)
        next_token_logits[~torch.isin(next_token_logits, top_k_values)] = float('-inf')
        next_token_prob = torch.softmax(next_token_logits, dim=-1)

        next_token_id = torch.multinomial(next_token_prob, num_samples=1)

        next_token = tokenizer.decode(next_token_id[0])
        print(f"Next predicted token: {next_token}")

        if next_token_id.item() == tokenizer.eos_token_id:
            log(f'task complete {task_id}: EOS token')
            server.task_complete(task_id)
        
        return next_token_id

HOST = '0.0.0.0'
PORT = 19999

import datetime

def log(message):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    log(f"{HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        with conn:
            data = receive_data(conn)
            if data is not None:
                task_id = data['task_id']
                task_id = f'{addr[0]}/{task_id}'
                log(f'working task {task_id}')
                if 'feature' in data:
                    feature = data['feature'].to(device)
                    log(f'request receive {feature.shape}')
                    outputs = generate(feature, task_id)
                    log(f'inference complete')
                    send_data(conn, outputs.cpu())
                    log(f'send tensor {outputs.shape}')
                else:
                    log(f'task complete {task_id}: client stops generation')
                    server.task_complete(task_id)
            conn.close()
