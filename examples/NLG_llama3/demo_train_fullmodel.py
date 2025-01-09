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
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaModel,
    LlamaConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm  
import time  
from utils import load_pretrain, print_model_stats, print_trainable_parameters

#================================model init =================================
model_name = "/mnt/data/zyx/llama-models/models/llama3/Meta-Llama-3-8B-Instruct"
dataset_name = "/home/yjh/splitfed-lora/examples/NLG_llama3/guanaco-llama3-1k"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = AutoModelForCausalLM.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
).to('cuda')
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=2,
    lora_alpha=4,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"]
)
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True


dataset = load_dataset(dataset_name, split = "train")
train_dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)


num_epochs = 5
learning_rate = 2e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#==============================training========================
global_step = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True,max_length=512).to('cuda:0')
        optimizer.zero_grad()
        outputs = model(**inputs, labels = inputs['input_ids'])
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch: {epoch + 1}, Average Loss: {avg_loss:.4f}")
