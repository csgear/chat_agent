import json
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


with open("../dataset/conversation_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)["conversations"]
        token = tokenizer(str(line), add_special_tokens=False)
        input_ids = token["input_ids"]
        attention_mask = token["attention_mask"]





