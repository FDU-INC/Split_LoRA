# 1.Environment Requirements
## 1.1 Environment Requirements

We have verified in the environment below:

+ OS: Ubuntu 18.04 

+ Python: 3.8.19

|    |torch <br> 2.4.1+cu121  |  transformers<br>4.46.1 | spacy | tqdm | tensorboard|progress|
|---|:---:|:---:|---|---|---|---|

<i>Note: You still need the original pre-trained checkpoint from [Hugging Face](https://huggingface.co/) to use the LoRA checkpoints.</i>

## 1.2Quick Start
1. Install Package
```Shell
conda create -n llama3 python=3.8.19 -y
conda activate llama3
```
2. Llama3 split inference
```shell
python examples/NLG_llama3/demo_infer_splitmodel_loadpretrain.py
```
or you can choose your device
```shell
CUDA_VISIBLE_DEVICES=0 python examples/NLG_llama3/demo_infer_splitmodel_loadpretrain.py
```
3. Llama3 split training
```shell
python examples/NLG_llama3/demo_train_Splitmodel.py
```
or you can choose your device
```shell
CUDA_VISIBLE_DEVICES=0 python examples/NLG_llama3/demo_train_Splitmodel.py
```

