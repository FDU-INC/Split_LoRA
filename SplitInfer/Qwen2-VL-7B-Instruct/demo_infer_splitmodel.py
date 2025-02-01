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
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'# Set the visible CUDA devices (GPUs) for PyTorch

import torch
from torch import nn
import torch.nn.functional as F

from transformers import  AutoProcessor,Qwen2VLConfig,AutoTokenizer
from qwen_vl_utils import process_vision_info
from Qwen2vl_modelsplit_splitLora import Qwen2VLForConditionalGeneration_Client, Qwen2VLForConditionalGeneration_Server

from utils import load_pretrained_Qwen2VL
#==============================model init=====================================
model_name = "/mnt/data/yjh/model/Qwen2-VL-7B-Instruct"
configuration = Qwen2VLConfig.from_pretrained(model_name)
configuration._attn_implementation = "flash_attention_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_client = Qwen2VLForConditionalGeneration_Client(configuration)
model_server = Qwen2VLForConditionalGeneration_Server(configuration)
lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias = False)
model_client, model_server, lm_head = load_pretrained_Qwen2VL(model_name, model_client, model_server, lm_head)
model_client = model_client.half().cuda(0)
model_server = model_server.half().cuda(1)
lm_head = lm_head.half().cuda(1)
#============================prepare for input================================
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
# 输入消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to('cuda:0')

#============================inference ================================
with torch.no_grad():
    generated_tokens = []
    max_length = 64
    for i in range(max_length):
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to('cuda:0')
        hidden_states, causal_mask, position_ids = model_client(**inputs)
        hidden_states = hidden_states.cuda(1)
        position_ids = position_ids.cuda(1)
        outputs = model_server(hidden_states=hidden_states, causal_mask=causal_mask, position_ids=position_ids)
        logits = lm_head(outputs[0])
        last_token_logits = logits[:, -1, :]
        softmax_logits = F.softmax(last_token_logits, dim = -1)
        predicted_token_id = torch.argmax(softmax_logits, dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        text += predicted_token
        generated_tokens.append(predicted_token)
    final_answer = ''.join(generated_tokens)
    print(final_answer)