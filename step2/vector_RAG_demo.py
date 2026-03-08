from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型加载（单例模式优化）
model_name = "Qwen/Qwen3-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_tokenizer():
    if not hasattr(get_model_tokenizer, "model"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        get_model_tokenizer.model = model
        get_model_tokenizer.tokenizer = tokenizer
    return get_model_tokenizer.model, get_model_tokenizer.tokenizer


def generate_response(prompt):
    model, tokenizer = get_model_tokenizer()

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=312
    )

    # 过滤输入部分并解码 [[6]]
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def customer_chat(query, informations):
    # 构建RAG增强Prompt
    system_prompt = (
        f"""
        你是一个跨境智能客服服务器，你现在需要回答用户的提问，使用如下知识和内容:'{informations}'
        1、你无需思考，只需要按提供给你的知识回答即可！
        2、你在回答问题时只能从给你的知识和内容中进查找，如果找不到对应的内容，就必须回答'找不到'。
        3、你不能回答给你的参考资料之外的内容。
        """
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    tokenizer = get_model_tokenizer()[1]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 生成完整响应
    return generate_response(prompt)



# 使用示例
if __name__ == "__main__":

    import vector_demo

    embedding_model = vector_demo.EmbeddingModel()
    with open("phone.txt", "r", encoding="utf-8") as f:
        content = f.read()
    informations = embedding_model.split_text_by_word_count(content,max_word_count=256)

    # 模拟RAG检索到的信息
    queries = [
        "手机的地域范围是哪里？",
        "我在什么情况下可以免费手机更换？",
        "手机的质保期是多久？"
    ]

    for query in queries:
        sorted_texts,sorted_scores = embedding_model.search_similar(query, informations)
        response = customer_chat(query, sorted_texts[:3])
        print(f"Q: {query}\nA: {response}\n{'=' * 40}")
