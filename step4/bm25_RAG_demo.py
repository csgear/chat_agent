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

from rank_bm25 import BM25Okapi
#query是需要查询的文本，documents为文本库，top_n为返回最接近的n条文本内容
def get_top_n_sim_text(query: str, documents,top_n = 3):
    tokenized_corpus = []
    for doc in documents:
        text = []
        for char in doc:
            text.append(char)
        tokenized_corpus.append(text)

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = [char for char in query]
    #doc_scores = bm25.get_scores(tokenized_query)  # array([0., 0.93729472, 0.])

    results = bm25.get_top_n(tokenized_query, tokenized_corpus, n=top_n)

    results = ["".join(res) for res in results]
    return results


def split_text_by_word_count(text, max_word_count):
    """
    将文本按字数分段

    参数:
        text (str): 要分段的文本内容
        max_word_count (int): 每段的最大字数

    返回:
        list: 分段后的文本列表
    """
    words = text.split()  # 按空格分割单词
    segments = []
    current_segment = []
    current_word_count = 0

    for word in words:
        # 添加当前单词到当前段，并更新字数统计
        current_segment.append(word)
        current_word_count += len(word) + 1  # +1 是为了考虑单词之间的空格

        # 如果当前段字数超过限制，则结束当前段并开始新段
        if current_word_count > max_word_count:
            # 如果当前段只有一个单词且超过限制，则单独处理（防止长单词无法放入）
            if len(current_segment) == 1:
                segments.append(current_segment[0])
                current_segment = []
                current_word_count = 0
            else:
                # 回退到上一个单词，完成当前段
                segments.append(' '.join(current_segment[:-1]))
                current_segment = [word]
                current_word_count = len(word) + 1

    # 添加最后一个段
    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

# 使用示例
if __name__ == "__main__":

    with open("phone.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # 模拟RAG检索到的信息
    queries = [
        "手机的地域范围是哪里？",
        "我在什么情况下可以免费手机更换？",
        "手机的质保期是多久？"
    ]

    for query in queries:
        informations = split_text_by_word_count(content, max_word_count=128)
        bm25_results = get_top_n_sim_text(query, informations)
        response = customer_chat(query, bm25_results)
        print(f"Q: {query}\nA: {response}\n{'=' * 40}")


