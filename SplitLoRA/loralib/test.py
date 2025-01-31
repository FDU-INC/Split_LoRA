from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
print(ds)