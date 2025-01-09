#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU_ISI
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU_ISI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
from transformers import AutoTokenizer
import transformers
import torch
from torch import nn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaModel,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TextStreamer,
    Trainer
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers.models.llama.modeling_llama import LlamaForCausalLM #, LlamaModel_Client
from modelsplit import LlamaModel_Client, LlamaModel_Server
import os, wandb

from utils import load_pretrain

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = "/mnt/data/zyx/llama-models/models/llama3/Meta-Llama-3-8B-Instruct"
configuration = LlamaConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


model1 = LlamaForCausalLM.from_pretrained(model_name).cuda()

# model2 = LlamaModel(configuration)
# lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
# model2,lm_head = load_pretrain(model2,model_name,lm_head)

# input_sentence = "How are you"
# input_sentence = "How are you doing today"
input_sentence = "Who is Crayon Shinchan?\n"
print(f"Input Sentence: {input_sentence}")

inputs = tokenizer(input_sentence, return_tensors='pt')

model1.eval()
# model2.eval()
# lm_head.eval()

print("Split inference token by token:")
with torch.no_grad():
    for i in range(30):
        outputs1 = model1(**inputs)
        logits1 = outputs1.logits
        last_token_logits1 = logits1[:, -1, :]
        predicted_token_id1 = torch.argmax(last_token_logits1, dim=-1).item()
        predicted_token1 = tokenizer.decode(predicted_token_id1)
        
        # import pdb; pdb.set_trace()
        input_sentence = input_sentence + predicted_token1
        print(input_sentence)
        inputs = tokenizer(input_sentence, return_tensors='pt')
        
        
    
# import pdb; pdb.set_trace()

# tokenizer.decode(torch.argmax(logits1[:, 2, :], dim=-1).item())

# with torch.no_grad():
#     outputs1 = model1(**inputs)
#     logits1 = outputs1.logits
#     last_token_logits1 = logits1[:, -1, :]
#     predicted_token_id1 = torch.argmax(last_token_logits1, dim=-1).item()
#     predicted_token1 = tokenizer.decode(predicted_token_id1)
    
# print(predicted_token1)
# import pdb; pdb.set_trace()
"""
with torch.no_grad():
    outputs2 = model2(**inputs)
    logits2 = lm_head(outputs2.last_hidden_state)
    last_token_logits2 = logits2[:, -1, :]
    predicted_token_id2 = torch.argmax(last_token_logits2, dim=-1).item()
    predicted_token2 = tokenizer.decode(predicted_token_id2)
    
print(f"Input Sentence: {input_sentence}")
print(f"Predicted Next Token by model1: {predicted_token1}")
print(f"Predicted Next Token by model2: {predicted_token2}")
"""

# lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
"""
#jetson

with torch.no_grad():
    hidden_states, causal_mask, position_ids = model_client(**inputs)

# PC
outputs = model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)

# import pdb; pdb.set_trace()
lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
logits = lm_head(outputs[0])
last_token_logits = logits[:, -1, :]
predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
predicted_token = tokenizer.decode(predicted_token_id)
print(f"Input Sentence: {input_sentence}")
print(f"Predicted Next Token: {predicted_token}")
# import pdb; pdb.set_trace()
"""