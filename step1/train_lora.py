import json
import torch
from tqdm import tqdm
from datasets import Dataset
#pip install transformers==4.32.1
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
tokenizer.pad_token = tokenizer.eos_token

# 定义LoRA配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],  # 覆盖所有注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # 保存关键模块的完整参数
)

# 应用LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 数据处理函数
def format_conversation(example):
    """带数据校验的对话格式处理"""
    messages = []
    for msg in example["conversations"]:
        # 自动修正常见键名错误
        role = msg.get("role", msg.get("assistant", "unknown")).lower()
        content = msg.get("content", "")

        # 验证角色有效性
        if role not in ["user", "assistant"]:
            if len(messages) == 0:
                role = "user"  # 第一条默认为user
            else:
                role = "assistant" if messages[-1]["role"] == "user" else "user"

        # 跳过无效消息
        if len(content.strip()) < 1:
            continue

        messages.append({"role": role, "content": content})

    # 构建对话模板（增加容错）
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        print(f"Template error: {e}")
        return None

    # 生成labels（带自动对齐）
    labels = []
    current_role = None
    for msg in messages:
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        if msg["role"] == "assistant":
            labels.extend(content_ids + [tokenizer.eos_token_id])
            current_role = "assistant"
        else:
            labels.extend([-100] * (len(content_ids) + 1))  # +1对应eos_token
            current_role = "user"

    return {"text": text, "labels": labels[:max_length]} if text else None

# 加载数据集
def load_dataset(file_path):
    data = []
    error_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f)):
            try:
                # 原始数据加载
                raw_data = json.loads(line)

                # 数据格式转换
                corrected_convs = []
                for msg in raw_data["conversations"]:
                    # 自动修正键名
                    new_msg = {
                        "role": msg.get("role", msg.get("assistant", "user")),
                        "content": msg.get("content", "")
                    }
                    # 修正角色值
                    if new_msg["role"] not in ["user", "assistant"]:
                        new_msg["role"] = "assistant" if len(corrected_convs) > 0 and corrected_convs[-1][
                            "role"] == "user" else "user"
                    corrected_convs.append(new_msg)

                # 处理修正后的对话
                formatted = format_conversation({"conversations": corrected_convs})
                if formatted and len(formatted["text"]) > 10:
                    data.append(formatted)
                else:
                    print(f"跳过无效对话：第{line_idx + 1}行")

            except Exception as e:
                error_count += 1
                print(f"错误处理第{line_idx + 1}行：{str(e)}")
                if error_count > 10:
                    raise RuntimeError("发现过多错误，请先修正数据格式")

    print(f"成功加载{len(data)}条有效数据（跳过{error_count}条无效数据）")
    return Dataset.from_list(data)

dataset = load_dataset("../dataset/conversation_dataset.jsonl")

# 数据集预处理
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 对齐labels
    labels = torch.full(
        (len(examples["text"]), max_length),
        -100,
        dtype=torch.long
    )
    for i, lbl in enumerate(examples["labels"]):
        labels[i, :len(lbl)] = torch.LongTensor(lbl[:max_length])

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=["text", "labels"]
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8
)

# 训练参数
# 调整训练参数（适用于24G显存）
training_args = TrainingArguments(
    per_device_train_batch_size=6,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_steps=10,
    report_to="none",
    output_dir=output_dir,
    save_safetensors=True,  # 启用安全格式保存
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存最终模型
# 训练结束后使用PEFT的保存方法
trainer.model.save_pretrained(
    output_dir,
    safe_serialization=True,  # 自动处理共享张量
    save_embedding_layers=True  # 如果需要保留embedding层参数
)


