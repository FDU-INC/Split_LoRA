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
import glob
from safetensors.torch import load_file

def display_model_statistics(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024**2)
    print(type(model).__name__)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

    return

def output_dict(file_name, dict):
    import os
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    sorted_keys = sorted(dict.keys())
    with open(file_name, 'w') as f:
        for idx, key in enumerate(sorted_keys):
            #f.write(f"{key}: {dict[key]}\n")
            if idx < 10:
                f.write(f"{key}: {dict[key]}\n")
            else:
                f.write(key + '\n')
            
    f.close()

def load_pretrained_Qwen2VL(model_path, model_client, model_server, lm_head):
    file_paths = glob.glob(os.path.join(model_path, "*.safetensors"))
    file_paths.sort(key = lambda x: int (os.path.basename(x).split('-')[1]))
    pretrained_state_dict = {}
    for filename in file_paths:
        loaded_state_dict = load_file(filename)
        pretrained_state_dict.update(loaded_state_dict)

    visual_update_dict = {}
    model_client_update_dict = {}
    model_server_update_dict = {}

    for key, value in pretrained_state_dict.items():
        if key == "lm_head.weight":
            lm_head.weight.data = value
            continue
        model_type = key.split(".")[0]
        if model_type == "model":
            if key == "model.embed_tokens.weight":
                model_client_update_dict[key] = value
            elif key == "model.norm.weight":
                model_server_update_dict[key] = value
            else:
                layer_num = int(key.split('.')[2])
                if layer_num < 7:
                    #print("client layer", layer_num)
                    model_client_update_dict[key] = value
                else:
                    new_layer_num = layer_num - 7
                    new_key = key.replace(f'layers.{layer_num}', f'layers.{new_layer_num}')
                    #print("server layer", layer_num)
                    model_server_update_dict[new_key] = value      
        elif model_type == "visual":
            visual_update_dict[key] = value

    model_client.load_state_dict(model_client_update_dict, strict = False)
    model_client.load_state_dict(visual_update_dict, strict = False)
    model_server.load_state_dict(model_server_update_dict, strict = False)

    return model_client, model_server, lm_head

def calculate_tensor_size_mb(tensor, name):
    # 检查张量是否为 None
    if tensor is None:
        print(f"错误: 张量 '{name}' 是 None.")
        return None
    # 获取张量的元素数量
    num_elements = tensor.numel()
    
    # 获取元素的字节大小
    element_size = tensor.element_size()  # 以字节为单位
    
    # 计算总大小（字节）
    total_size_bytes = num_elements * element_size
    
    # 转换为 MB（1 MB = 1024 * 1024 字节）
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # 打印张量的名称和大小
    print(f"张量 '{name}' 大小: {total_size_mb:.2f} MB")
    return total_size_mb