import json
import torch
import safetensors.torch as safetorch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 配置参数
model_name = "Qwen/Qwen3-1.7B"
output_dir = "qwen_lora_finetuned"
max_length = 384  # 根据显存调整

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False  # 梯度检查点需要关闭cache
)
print("base_model load successful!")
from peft import PeftModel
model = PeftModel.from_pretrained(model, "qwen_lora_finetuned")

messages = [
    {"role": "user", "content": "我没收到货呢所以就想知道是哪里的问题?"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=max_length
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

