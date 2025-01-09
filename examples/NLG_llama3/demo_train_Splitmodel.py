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
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter  
from modelsplit import LlamaModel_Client, LlamaModel_Server
from utils import load_pretrain, load_pretrain_split, print_model_stats

#==========================model init =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #your device, such as "0" or "0, 1, 2"
model_name = "/mnt/data/zyx/llama-models/models/llama3/Meta-Llama-3-8B-Instruct" #your model path
dataset_name = "/home/yjh/splitfed-lora/examples/NLG_llama3/guanaco-llama3-1k" #your dataset path

#config and tokenizer
configuration = LlamaConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

model_client = LlamaModel_Client(configuration)
model_server = LlamaModel_Server(configuration)
lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
model_client, model_server, lm_head = load_pretrain_split(model_client, model_server, lm_head, model_name)

model_client = model_client.half().to("cuda")
model_server = model_server.half().to("cuda")
lm_head = lm_head.half().to("cuda")

#Lora config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"]
)
model_client = get_peft_model(model_client, peft_config)
model_server = get_peft_model(model_server, peft_config)

dataset = load_dataset(dataset_name, split="train")
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

num_epochs = 5  # 增加到 5 个 epoch
learning_rate = 2e-4
optimizer = torch.optim.AdamW(list(model_client.parameters()) + list(model_server.parameters()), lr=learning_rate)
#========================training========================
for epoch in range(num_epochs):
    model_client.train()
    model_server.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc = f"Epoch {epoch + 1}/{num_epochs}")):
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
        optimizer.zero_grad()
        hidden_states, causal_mask, position_ids = model_client(**inputs)
        outputs = model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
        logits = lm_head(outputs[0])
        loss_fct = nn.CrossEntropyLoss()
        labels = inputs['input_ids'][:, 1:]  # 右移，去掉第一个 token 
        logits = logits[:, :-1, :]  # 去掉最后一个 logit
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        print(f"loss : {loss.item()}")
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch: {epoch + 1}, Average Loss: {avg_loss:.4f}")

# 保存微调后的模型
model_client.save_pretrained("./model/lora_model_client")
model_server.save_pretrained("./model/lora_model_server")