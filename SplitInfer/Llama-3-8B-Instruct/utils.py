#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU_ISI
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU_ISI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
import torch
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from safetensors.torch import load_file


def load_multiple_safetensors(filenames):
    #import pdb; pdb.set_trace()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_state_dict = {}
    for filename in filenames:
        loaded_state_dict = load_file(filename)
        combined_state_dict.update(loaded_state_dict)
    return combined_state_dict



def load_pretrain(model, lm_head, model_name):
    file_paths = [
    f"{model_name}/model-00001-of-00004.safetensors",
    f"{model_name}/model-00002-of-00004.safetensors",
    f"{model_name}/model-00003-of-00004.safetensors",
    f"{model_name}/model-00004-of-00004.safetensors"
]   
    pretrained_state_dict = load_multiple_safetensors(file_paths)
    model_dict = model.state_dict()
    state_dict = {}
    for key,value in pretrained_state_dict.items():
        new_key = key.replace('model.','')
        state_dict[new_key] = value

    lm_head.weight.data = pretrained_state_dict['lm_head.weight'].to(torch.float32)
    model_dict.update(state_dict)
    model.load_state_dict(state_dict, strict=False)
    #model = model.to(device)
    #import pdb;pdb.set_trace()

    return model,lm_head



def load_pretrain_split(client_model, server_model, lm_head, model_name):
    
    file_paths = [
    f"{model_name}/model-00001-of-00004.safetensors",
    f"{model_name}/model-00002-of-00004.safetensors",
    f"{model_name}/model-00003-of-00004.safetensors",
    f"{model_name}/model-00004-of-00004.safetensors"
]   
    pretrained_state_dict = load_multiple_safetensors(file_paths)
    client_dict = client_model.state_dict()
    server_dict = server_model.state_dict()

    client_update_dict = {}
    server_update_dict = {}
    state_dict = {}
    for key, value in pretrained_state_dict.items():
        new_key = key.replace('model.', '')
        state_dict[new_key] = value

    for key, value in state_dict.items():
        if any(f'layers.{i}' in key for i in range(8,32)) or key == 'norm.weight': 
            if 'layers.' in key:
                layer_num = int(key.split('.')[1])
                if layer_num >= 8 and layer_num <= 31:
                    new_layer_num = layer_num - 8
                    new_key = key.replace(f'layers.{layer_num}', f'layers.{new_layer_num}')
                else:
                    new_key = key.replace('model.', '')
            else:
                new_key = key.replace('model.', '')
            server_update_dict[new_key] = value
        elif any(f'layers.{i}' in key for i in range(8)) or key =='embed_tokens.weight':
            new_key = key.replace('model.','')
            client_update_dict[new_key] = value
        else:  
            #print(key)
            pass
    #update params
    client_dict.update(client_update_dict)
    client_model.load_state_dict(client_update_dict, strict=False)
    server_dict.update(server_update_dict)
    server_model.load_state_dict(server_update_dict, strict=False)
    lm_head.weight.data = pretrained_state_dict['lm_head.weight'].to(torch.float32)
    #import pdb; pdb.set_trace()
    return client_model, server_model,lm_head

# 打印模型参数信息
def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # 假设每个参数占用 4 字节（float32）

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

# 计算训练参数量
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"训练参数量 : {trainable_params} || 总的参数量 : {all_param} || 训练参数量占比%: {100 * (trainable_params / all_param):.2f}"
    )

def combined_fed_avg(clients, servers):
    """对客户端和服务器模型进行参数平均"""
    averaged_client_state_dict = {}
    averaged_server_state_dict = {}

    # 聚合客户端模型
    for key in clients[0].state_dict().keys():
        averaged_client_state_dict[key] = torch.mean(torch.stack([client.state_dict()[key] for client in clients]), dim=0)

    # 聚合服务器模型
    for key in servers[0].state_dict().keys():
        averaged_server_state_dict[key] = torch.mean(torch.stack([server.state_dict()[key] for server in servers]), dim=0)

    return averaged_client_state_dict, averaged_server_state_dict
