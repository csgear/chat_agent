import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import torch

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="TencentBAC/Conan-embedding-v1",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./embeddings_cache"
        )

    def embed_query(self, query):
        embedding = self.embeddings.embed_query(query)
        return np.array(embedding)

    def embed_querys(self, query_list):
        embedding_list = []
        for query in query_list:
            embedding = self.embed_query(query)
            embedding_list.append(embedding)
        return np.array(embedding_list)

    def search_similar(self, query, text_list):
        """
        查找与查询最相似的文本列表，按相似度降序排列

        参数:
            query (str): 输入查询
            embedding_list (np.array): 预计算的嵌入向量列表（n_samples, embedding_dim）
            text_list (list): 与嵌入对应的原始文本列表

        返回:
            tuple: (排序后的文本列表, 对应的相似度分数列表)
        """
        # 1. 生成查询嵌入
        query_emb = self.embed_query(query)  # [[3]] 使用Hugging Face嵌入模型
        embedding_list = self.embed_querys(text_list)

        # 2. 计算余弦相似度（通过点积实现，因已归一化）
        similarities = np.dot(embedding_list, query_emb)

        # 3. 按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]  # 降序排列

        # 4. 返回排序结果
        sorted_texts = [text_list[i] for i in sorted_indices]
        sorted_scores = similarities[sorted_indices]

        return sorted_texts, sorted_scores



    def split_text_by_word_count(self,text, max_word_count):
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


if __name__ == '__main__':

    embedding_model = EmbeddingModel()

    informations = [
        "手机的地域范围是哪里？",
        "我在什么情况下可以免费手机更换？",
        "手机的质保期是多久？"
    ]

    # 查询示例
    query = "手机保修多久？"
    sorted_texts, sorted_scores = embedding_model.search_similar(query, informations)

    print("排序结果：")
    for text, score in zip(sorted_texts, sorted_scores):
        print(f"相似度 {score:.4f} : {text}")
