from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# 配置参数
model_name = "Qwen/Qwen3-1.7B"
output_dir = "qwen_lora_finetuned"
max_length = 256  # 根据显存调整

# ── 数据文件配置 ────────────────────────────────────────────────
TRAIN_FILE = "../E-commerce dataset/dev.txt"
EVAL_FILE = "../E-commerce dataset/test.txt"
# ────────────────────────────────────────────────────────────────

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,  # 梯度检查点需要关闭cache
)
tokenizer.pad_token = tokenizer.eos_token

# 梯度检查点需要在PEFT包装前启用input grad钩子
model.enable_input_require_grads()

# 定义LoRA配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],  # 覆盖所有注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
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


def load_ecommerce_txt(file_path):
    """加载 E-commerce .txt 格式: label\tutt1\tutt2\t...\tresponse
    只保留 label=1 的正样本，将 utterances 映射为 user/assistant 交替对话。"""
    data = []
    skipped = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                skipped += 1
                continue
            utterances = [p.replace(" ", "") for p in parts[1:]]  # 去除分词空格
            # 奇数索引=user，偶数索引(含最后response)=assistant
            convs = []
            for i, utt in enumerate(utterances):
                convs.append({"role": "user" if i % 2 == 0 else "assistant", "content": utt})
            formatted = format_conversation({"conversations": convs})
            if formatted and len(formatted["text"]) > 10:
                data.append(formatted)
            else:
                skipped += 1
    print(f"加载 {len(data)} 条正样本（跳过 {skipped} 条）")
    return Dataset.from_list(data)


def load_dataset(file_path):
    """根据后缀自动选择加载方式"""
    if file_path.endswith(".txt"):
        return load_ecommerce_txt(file_path)
    # 原 JSONL 格式
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


dataset = load_dataset(TRAIN_FILE)
eval_data = load_dataset(EVAL_FILE)

# 数据集预处理


def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    # 对齐labels
    all_labels = []
    for lbl in examples["labels"]:
        padded = lbl[:max_length] + [-100] * (max_length - min(len(lbl), max_length))
        all_labels.append(padded)

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": all_labels
    }


processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=["text", "labels"]
)

processed_eval = eval_data.map(
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

# 训练参数（适用于16G显存）
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # 等效batch_size=16
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_steps=10,
    eval_strategy="epoch",          # 每epoch评估一次，用于比较dev/train效果
    report_to="none",
    output_dir=output_dir,
    dataloader_pin_memory=False,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_eval,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存最终模型
# 训练结束后使用PEFT的保存方法
trainer.model.save_pretrained(
    output_dir,
    safe_serialization=True
)
