#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU_ISI
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU_ISI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
import os
import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from modelsplit import LlamaModel_Client, LlamaModel_Server
from utils import load_pretrain_split, combined_fed_avg
#=========================model init =============================
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_name = "/mnt/data/zyx/llama-models/models/llama3/Meta-Llama-3-8B-Instruct"
dataset_name = "/home/yjh/splitfed-lora/examples/NLG_llama3/guanaco-llama3-1k"
dataset = load_dataset(dataset_name, split="train")
configuration = LlamaConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

num_clients = 3
clients = [LlamaModel_Client(configuration) for _ in range(num_clients)]
servers = [LlamaModel_Server(configuration) for _ in range(num_clients)]
lm_heads = [nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False).half() for _ in range(num_clients)]

for i in range(num_clients):
    clients[i], servers[i], lm_heads[i] = load_pretrain_split(clients[i], servers[i], lm_heads[i], model_name)
    clients[i].half()
    servers[i].half()
    lm_heads[i].half()

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"]
)
for i in range(num_clients):
    clients[i] = get_peft_model(clients[i], peft_config)
    servers[i] = get_peft_model(servers[i], peft_config)

total_size = len(dataset)
client_sizes = [total_size // num_clients] * num_clients
for i in range(total_size % num_clients):
    client_sizes[i] += 1
client_datasets = torch.utils.data.random_split(dataset, client_sizes)
train_dataloaders = [DataLoader(client_datasets[i], batch_size=1, shuffle=True) for i in range(num_clients)]

num_epochs = 100
learning_rate = 2e-4
optimizers = [torch.optim.AdamW(list(clients[i].parameters()) + list(servers[i].parameters()), lr=learning_rate) for i in range(num_clients)]

#=======================training=============================
for epoch in range(num_epochs):
    total_losses = 0.0
    for client_idx in range(num_clients):
        client = clients[client_idx].to('cuda')
        server = servers[client_idx].to('cuda')
        lm_head = lm_heads[client_idx].to('cuda')
        client.train()
        server.train()
        dataloader = train_dataloaders[client_idx]
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Client {client_idx + 1} Progress")):
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
            optimizers[client_idx].zero_grad()
            hidden_states, causal_mask, position_ids = client(**inputs)
            outputs = server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
            logits = lm_head(outputs[0])
            loss_fct = nn.CrossEntropyLoss()
            labels = inputs['input_ids'][:, 1:]
            logits = logits[:, :-1, :]
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_losses += loss.item()
            loss.backward()
            optimizers[client_idx].step()
            torch.cuda.empty_cache()
        client.to('cpu')
        server.to('cpu')
        lm_head.to('cpu')
        print(f"Client {client_idx + 1} training completed.")
    print("Aggregating model parameters...")
    avg_client_state_dict, avg_server_state_dict = combined_fed_avg(clients, servers)
    for client in clients:
        client.load_state_dict(avg_client_state_dict)
    for server in servers:
        server.load_state_dict(avg_server_state_dict)
    avg_loss = total_losses / len(dataset)
    print(f"Epoch: {epoch + 1}, Average Loss: {avg_loss:.4f}")

print("Saving fine-tuned models...")
for i in range(num_clients):
    clients[i].save_pretrained(f"./model/lora_model_client_{i}")
    servers[i].save_pretrained(f"./model/lora_model_server_{i}")