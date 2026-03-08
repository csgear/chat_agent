import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置参数
model_name = "Qwen/Qwen3-1.7B"
max_length = 512  # 根据显存调整

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 加载微调后的模型
model = PeftModel.from_pretrained(base_model, "qwen_lora_finetuned")
model = model.merge_and_unload()  # 合并LoRA权重方便推理 [[6]]
model.eval()
print("模型加载完成！")

def chat():
    """修复版多轮对话交互函数"""
    messages = []

    # Qwen2.5的专用系统提示写法
    system_prompt = {
        "role": "system",
        "content": "You are a professional e-commerce customer service assistant. First, you need to detect the language type that the other party is using. After analyzing the issue, please respond in a concise and friendly manner using the very language they've employed."
    }
    messages.append(system_prompt)

    while True:
        try:
            user_input = input("\n用户：").strip()
            if not user_input:
                continue
            if user_input.lower() in ["退出", "exit", "quit"]:
                break

            # 添加用户消息（自动过滤空内容）
            messages.append({"role": "user", "content": user_input})

            # 生成符合Qwen2.5格式的对话模板（关键修复点）[[9]]
            text = tokenizer.apply_chat_template(
                conversation=messages,  # 包含当前输入
                add_generation_prompt=True,
                tokenize=False
            )

            # 编码输入（增加长度校验）[[3]]
            model_inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length - 100,  # 为生成留出空间
                truncation=True,
                padding="max_length"
            ).to(model.device)

            # 生成回复（优化生成参数）[[6]]
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.85,
                    repetition_penalty=1.15,
                    eos_token_id=tokenizer.eos_token_id
                )

            # 精确截取新生成内容（关键修复）[[10]]
            input_len = model_inputs.input_ids.shape[1]
            response = tokenizer.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True
            ).strip()
            print(f"\n助手：{response}")
            # 添加助手回复到历史（带智能截断）
            messages.append({"role": "assistant", "content": response})


            # 自动维护对话历史（保留最近3轮+系统提示）[[8]]
            max_history = 6  # 3轮对话（user+assistant为一轮）
            if len(messages) > max_history + 1:  # +1是系统提示
                messages = [messages[0]] + messages[-max_history:]

        except KeyboardInterrupt:
            print("\n对话已终止")
            break
        except Exception as e:
            print(f"生成出错：{str(e)}")
            messages.pop()  # 移除当前错误输入

if __name__ == "__main__":
    print("欢迎使用客服助手！输入「退出」结束对话")
    chat()



