import copy
import re
from typing import Callable
import numpy as np

def check_has_chinese(text):
    """
    如果存在中文，返回True，否则返回False
    :param text:
    :return:  bool
    """
    if not isinstance(text, str):
        return False
    res = re.findall('[\u4e00-\u9fa5]+', text)
    return bool(res)

class TopwordEnhancement:
    def __init__(
            self,
            client,
            openai_model: str = "gpt-3.5-turbo",
            max_context_length: int = 4000,
            openai_model_temperature: float = 0.5,
            basic_model_instruction: str = "",
            corpus_instruction: str = "",
            embedder=None
    ) -> None:
        """
        定义了一个名为 TopwordEnhancement 的类，主要用于通过 OpenAI 的模型对给定的关键词或文档进行主题描述。

        Args:
            client：与 OpenAI API 的客户端连接。
            openai_model：指定使用的 OpenAI 模型，默认为 "gpt-3.5-turbo"。
            max_context_length：上下文的最大长度，默认为 4000。
            openai_model_temperature：模型的温度设置，默认为 0.5。
            basic_model_instruction：模型的基本指令。
            corpus_instruction：对语料库的指令，默认是空字符串。
            embedder: 实例化后的嵌入模型
        Returns:
            None
        """
        # do some checks on the input arguments
        assert openai_model is not None, "Please provide an openai model"
        assert max_context_length > 0, "Please provide a positive max_context_length"
        assert openai_model_temperature > 0, "Please provide a positive openai_model_temperature"

        self.client = client
        self.openai_model = openai_model
        self.max_context_length = max_context_length
        self.openai_model_temperature = openai_model_temperature
        if basic_model_instruction:
            self.basic_model_instruction_zh = basic_model_instruction
            self.basic_model_instruction_en = basic_model_instruction
        else:
            self.basic_model_instruction_zh = "你是一个乐于助人的助手，擅长从主题模型提取的关键词中推断出主题。你确保输出的所有内容都严格基于所提供的文本。使用中文回答问题。"
            self.basic_model_instruction_en = "You are a helpful assistant. You are excellent at inferring topics from top-words extracted via topic-modelling. You make sure that everything you output is strictly based on the provided text."
        self.corpus_instruction = f"下面是主题识别的语料库信息: {corpus_instruction}"
        self.embedder = embedder
        self.topic_name_description_prompt_function_zh = lambda tws: f"""我会提供一些词语，请提供这些词语的共同主题、描述。主题不超过5个字。
词语列表: {tws}
输出格式如下:
主题：xxx
描述：xxx
"""
        self.topic_name_description_prompt_function_en = lambda tws: f"""I'll provide some words. Please provide the common topic and descriptions for these words. topic should not exceed 5 words.
Word list: {tws}
The output format is as follows:
TOPIC: xxx
DESCRIPTION: xxx
"""

    def __str__(self) -> str:
        repr = f"TopwordEnhancement(openai_model = {self.openai_model})"
        return repr

    def __repr__(self) -> str:
        repr = f"TopwordEnhancement(openai_model = {self.openai_model})"
        return repr

    def count_tokens_api_message(self, messages: list[dict[str]]) -> int:
        """
        计算 API 消息中的 token 数量。
        Args:
            messages (list[dict[str]]): List of messages from the API.
        Returns:
            int: Number of tokens in the messages.
        """
        n_tokens = 0
        for message in messages:
            for key, value in message.items():
                if key == "content":
                    n_tokens += len(self.embedder.encoding_for_model(value))

        return n_tokens

    def describe_topic_topwords_completion_object(self,
                                                  topwords: list[str],
                                                  n_words: int = None,
                                                  query_function: Callable = lambda
                                                          tws: f"Please give me the common topic of those words: {tws}. Also describe the various aspects and sub-topics of the topic."):
        """
        根据给定的 topwords 使用 OpenAI 模型生成主题描述。
        Args:
            topwords：主题的 topwords 列表。
            n_words：用于查询的词数，如果为 None，则使用所有 topwords。
            query_function：生成查询内容的函数，默认为生成一个描述主题的查询。

        Returns:
            openai.ChatCompletion: A description of the topics by the model in the form of an OpenAI ChatCompletion object.
        """

        if n_words is None:
            n_words = len(topwords)

        if type(topwords) == dict:
            topwords = topwords[0]

        topwords = topwords[:n_words]
        topwords = np.array(topwords)

        print("开始计算上下文长度不超过llm长度")
        tokens_cumsum = np.cumsum([len(self.embedder.encoding_for_model(tw + ", ")) for tw in topwords]) + len(
            self.embedder.encoding_for_model(self.basic_model_instruction_zh + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print(
                "Too many topwords given. Using only the first part of the topwords that fits into the context length. Number of topwords used: ",
                np.argmax(tokens_cumsum > self.max_context_length))
            n_words = np.argmax(tokens_cumsum > self.max_context_length)
            topwords = topwords[:n_words]

        messages = [
            {"role": "system",
             "content": self.basic_model_instruction_zh + " " + self.corpus_instruction},
            {"role": "user", "content": query_function(topwords)},
        ]
        print(f"最终生成的messages是: {messages}")
        completion = self.client.chat.completions.create(model=self.openai_model,
                                                         messages=messages,
                                                         temperature=self.openai_model_temperature)
        return completion

    def describe_topic_documents_completion_object(self,
                                                   documents: list[str],
                                                   truncate_doc_thresh=100,
                                                   n_documents: int = None,
                                                   query_function: Callable = lambda
                                                           docs: f"Please give me the common topic of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic."):
        """
        根据给定的文档生成主题描述。
        Args:
            documents：文档的列表。
            truncate_doc_thresh：文档的长度阈值，超出此阈值的文档将被截断。
            n_documents：用于查询的文档数量，如果为 None，则使用所有文档。
            query_function：生成查询内容的函数。
        Returns:
            openai.ChatCompletion: A description of the topics by the model in the form of an openai.ChatCompletion object.
        """

        if n_documents is None:
            n_documents = len(documents)
        documents = documents[:n_documents]

        # prune documents based on number of tokens they contain 
        new_doc_lis = []
        for doc in documents:
            doc = doc.split(" ")
            if len(doc) > truncate_doc_thresh:
                doc = doc[:truncate_doc_thresh]
            new_doc_lis.append(" ".join(doc))
        documents = new_doc_lis

        # if too many documents are given, use only the first part of the documents that fits into the context length
        tokens_cumsum = np.cumsum([len(self.embedder.encoding_for_model(doc + ", ")) for doc in documents]) + len(
            self.embedder.encoding_for_model(self.basic_model_instruction_zh + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print(
                "Too many documents given. Using only the first part of the documents that fits into the context length. Number of documents used: ",
                np.argmax(tokens_cumsum > self.max_context_length))
            n_documents = np.argmax(tokens_cumsum > self.max_context_length)
            documents = documents[:n_documents]

        completion = self.client.chat.completions.create(model=self.openai_model,
                                                         messages=[
                                                             {"role": "system",
                                                              "content": self.basic_model_instruction_zh + " " + self.corpus_instruction},
                                                             {"role": "user", "content": query_function(documents)},
                                                         ],
                                                         temperature=self.openai_model_temperature)

        return completion

    @staticmethod
    def sample_identity(n_docs: int) -> np.ndarray:
        """
        生成一个包含文档索引的数组，保持文档的原始顺序。
        Args:
            n_docs (int): Number of documents.

        Returns:
            np.ndarray: An array containing document indices from 0 to (n_docs - 1).
        """

        return np.arange(n_docs)

    @staticmethod
    def sample_uniform(n_docs: int) -> np.ndarray:
        """
        随机抽样文档索引，返回打乱顺序的文档索引数组。
        Args:
            n_docs (int): Number of documents.

        Returns:
            np.ndarray: An array containing randomly permuted document indices from 0 to (n_docs - 1).
        """

        return np.random.permutation(n_docs)

    @staticmethod
    def sample_poisson(n_docs: int) -> np.ndarray:
        """
        根据 Poisson 分布随机抽样文档索引，使得列表前面的文档被选中的概率更高。
        Args:
            n_docs (int): Number of documents.

        Returns:
            np.ndarray: An array containing randomly permuted document indices, with more documents drawn from the beginning of the list.
        """

        return np.random.poisson(1, n_docs)

    def describe_topic_documents_sampling_completion_object(
            self,
            documents: list[str],
            truncate_doc_thresh=100,
            n_documents: int = None,
            query_function: Callable = lambda
                    docs: f"Please give me the common topic of the sample of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.",
            sampling_strategy: str = None, ):
        """
        描述主题时，根据给定的采样策略对文档进行采样。

        Args:
            documents (list[str]): List of documents ordered by similarity to the topic's centroid.
            truncate_doc_thresh (int, optional): Threshold for the number of words in a document. If a document exceeds this threshold, it is truncated. Defaults to 100.
            n_documents (int, optional): Number of documents to use for the query. If None, all documents are used. Defaults to None.
            query_function (Callable, optional): Function to query the model. Defaults to a lambda function generating a query based on the provided documents.
            sampling_strategy (Union[Callable, str], optional): Strategy to sample the documents. If None, the first provided documents are used.
                If it's a string, it's interpreted as a method of the class (e.g., "sample_uniform" is interpreted as self.sample_uniform). It can also be a custom sampling function. Defaults to None.

        Returns:
            openai.ChatCompletion: A description of the topic by the model in the form of an openai.ChatCompletion object.
        """

        if type(sampling_strategy) == str:
            if sampling_strategy == "topk":
                sampling_strategy = self.sample_identity
            if sampling_strategy == "identity":
                sampling_strategy = self.sample_identity
            elif sampling_strategy == "uniform":
                sampling_strategy = self.sample_uniform
            elif sampling_strategy == "poisson":
                sampling_strategy = self.sample_poisson

        new_documents = [documents[i] for i in sampling_strategy(n_documents)]

        result = self.describe_topic_documents_completion_object(new_documents, truncate_doc_thresh, n_documents,
                                                                 query_function)
        return result

    def describe_topic_document_sampling_str(
            self,
            documents: list[str],
            truncate_doc_thresh=100,
            n_documents: int = None,
            query_function: Callable = lambda
                    docs: f"Please give me the common topic of the sample of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.",
            sampling_strategy: str = None, ) -> str:
        """
        类似于 describe_topic_documents_sampling_completion_object，但返回主题描述的字符串。
        Describe a topic based on a sample of its documents by using the openai model.

        Args:
            documents (list[str]): List of documents ordered by similarity to the topic's centroid.
            truncate_doc_thresh (int, optional): Threshold for the number of words in a document. If a document exceeds this threshold, it is truncated. Defaults to 100.
            n_documents (int, optional): Number of documents to use for the query. If None, all documents are used. Defaults to None.
            query_function (Callable, optional): Function to query the model. Defaults to a lambda function generating a query based on the provided documents.
            sampling_strategy (Union[Callable, str], optional): Strategy to sample the documents. If None, the first provided documents are used.
                If it's a string, it's interpreted as a method of the class (e.g., "sample_uniform" is interpreted as self.sample_uniform). It can also be a custom sampling function. Defaults to None.

        Returns:
            str: A description of the topic by the model in the form of a string.
        """

        completion = self.describe_topic_document_sampling_completion_object(documents, truncate_doc_thresh,
                                                                             n_documents, query_function,
                                                                             sampling_strategy)
        return completion.choices[0].message.content

    def extract_topic_and_description(self, text, language="english"):
        if language == "chinese":
            topic_pattern = r"主题：\s*(.*)"
            description_pattern = r"描述：\s*(.*)"
        else:
            topic_pattern = r"TOPIC:\s*(.*)"
            description_pattern = r"DESCRIPTION:\s*(.*)"

        topic_match = re.search(topic_pattern, text)
        description_match = re.search(description_pattern, text, re.DOTALL)

        topic = topic_match.group(1) if topic_match else None
        description = description_match.group(1).replace("\n", "").strip() if description_match else None

        return topic, description
    def generate_topic_name_and_describe(self,
                                topwords: list[str],
                                n_words: int = None,
                                language: str = "english"
                                ) -> str:
        """
        生成一个主题名称。基于 topwords 和查询函数来生成主题名称。
        Args:
            topwords (list[str]): List of topwords.
            n_words (int, optional): Number of words to use for the query. If None, all words are used.
            language: 语言
        Returns:
            str: A topic name generated by the model in the form of a string.
        """
        if n_words is None:
            n_words = len(topwords)

        if type(topwords) == dict:
            topwords = topwords[0]

        topwords = topwords[:n_words]
        topwords = np.array(topwords)
        print("开始计算上下文长度不超过llm长度")
        tokens_cumsum = np.cumsum([len(self.embedder.encoding_for_model(tw + ", ")) for tw in topwords]) + len(
            self.embedder.encoding_for_model(self.basic_model_instruction_zh + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print(
                "Too many topwords given. Using only the first part of the topwords that fits into the context length. Number of topwords used: ",
                np.argmax(tokens_cumsum > self.max_context_length))
            n_words = np.argmax(tokens_cumsum > self.max_context_length)
            topwords = topwords[:n_words]
        topwords_str = ",".join(topwords)
        messages = [
            {"role": "system","content": self.basic_model_instruction_en + " " + self.corpus_instruction},
            {"role": "user", "content": self.topic_name_description_prompt_function_en(topwords_str)},
        ]
        max_retries = 10
        retries = 0
        all_errors = []
        error_messages = ""
        while retries < max_retries:
            try:
                new_messages = copy.deepcopy(messages)
                if error_messages and retries % 2 == 1:  # 奇数次的时候，加上错误日志
                    new_messages[-1]["content"] += f"\n{error_messages}"
                print(f"最终生成的messages是: {new_messages}")
                completion = self.client.chat.completions.create(model=self.openai_model,
                                                                 messages=new_messages,
                                                                 temperature=self.openai_model_temperature)
                output = completion.choices[0].message.content
                assert "TOPIC:" in output, "output does not contain 'TOPIC:'"
                assert "DESCRIPTION:" in output, "output does not contain 'DESCRIPTION:'"
                topic, description = self.extract_topic_and_description(text=output, language=language)
                assert topic, "TOPIC: format is incorrect"
                assert description, "DESCRIPTION: format is incorrect"
                return topic, description
            except Exception as e:
                retries += 1
                error_messages = f"Your last response was incorrect because {e}"
                all_errors.append(e)
                print(f"重试了{retries}次，还是没有输出符合要求,错误是:{error_messages}")
        return "未知主题", "未知描述"
