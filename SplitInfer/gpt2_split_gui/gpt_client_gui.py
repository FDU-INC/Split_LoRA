import tkinter as tk
import threading
import datetime

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from arch.arch import GPT2LMHeadModel
from arch.client import GPT2LMHeadModelClient

from network import *

model = GPT2LMHeadModel.from_pretrained('./gpt2-model')
client = GPT2LMHeadModelClient(model, 2)

tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-model', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

client.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client.to(device)

root = tk.Tk()
root.title("GPT")
root.geometry("800x600")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)
root.grid_columnconfigure(0, weight=1)

text_box = tk.Text(root, wrap="word")
text_box.grid(row=0, column=0, sticky="nsew")

default_text = "World's first 3D-printed hotel takes shape in Texas."
text_box.insert(tk.END, default_text)

def log(message):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)

def get_task_id():
    return 1

def generate(host, port):
    text = text_box.get(1.0, tk.END)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_text = text
    task_id = get_task_id()
    first_pred = True
    for _ in range(100):
        with torch.no_grad():
            feature = client(**inputs)
            if first_pred:
                first_pred = False
            else:
                feature = feature[:, -1, :].unsqueeze(0)
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                send_data(s, {
                    'task_id': task_id,
                    'feature': feature.cpu()
                })
                next_token_id = receive_data(s).to(device)

            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)
            ones_to_add = torch.ones(*inputs['attention_mask'].shape[:-1], 1).to(device)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], ones_to_add], dim=-1)

            next_token = tokenizer.decode(next_token_id[0])
            log(f"Next predicted token: {next_token}")
            if next_token_id.item() == tokenizer.eos_token_id:
                return
            generated_text += next_token
            root.after(0, update_text, generated_text)

    log(generated_text)
    root.after(0, update_text, generated_text)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        send_data(s, {
            'task_id': task_id,
        })

def update_text(text):
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, text)

# HOST = '192.168.0.100'
HOST = '127.0.0.1'
PORT = 19999

def insert_hello():
    gpt_thread = threading.Thread(target=generate, args=(HOST, PORT))
    gpt_thread.start()

button = tk.Button(root, text="Generate", command=insert_hello)
button.grid(row=1, column=0, sticky="ew")

root.mainloop()
