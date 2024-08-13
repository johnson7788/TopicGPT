from typing import Callable
import numpy as np

basic_instruction = "You are a helpful assistant. You are excellent at inferring topics from top-words extracted via topic-modelling. You make sure that everything you output is strictly based on the provided text."

class TopwordEnhancement:
    def __init__(
            self,
            client,
            openai_model: str = "gpt-3.5-turbo",
            max_context_length: int = 4000,
            openai_model_temperature: float = 0.5,
            basic_model_instruction: str = basic_instruction,
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
        self.basic_model_instruction = basic_model_instruction
        self.corpus_instruction = f"下面是主题识别的语料库信息: {corpus_instruction}"
        self.embedder = embedder

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

        # if too many topwords are given, use only the first part of the topwords that fits into the context length, 计算上下文长度不超过llm长度
        tokens_cumsum = np.cumsum([len(self.embedder.encoding_for_model(tw + ", ")) for tw in topwords]) + len(
            self.embedder.encoding_for_model(self.basic_model_instruction + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print(
                "Too many topwords given. Using only the first part of the topwords that fits into the context length. Number of topwords used: ",
                np.argmax(tokens_cumsum > self.max_context_length))
            n_words = np.argmax(tokens_cumsum > self.max_context_length)
            topwords = topwords[:n_words]

        completion = self.client.chat.completions.create(model=self.openai_model,
                                                         messages=[
                                                             {"role": "system",
                                                              "content": self.basic_model_instruction + " " + self.corpus_instruction},
                                                             {"role": "user", "content": query_function(topwords)},
                                                         ],
                                                         temperature=self.openai_model_temperature)

        return completion

    def describe_topic_topwords_str(self,
                                    topwords: list[str],
                                    n_words: int = None,
                                    query_function: Callable = lambda
                                            tws: f"Please give me the common topic of those words: {tws}. Also describe the various aspects and sub-topics of the topic. Make sure the descriptions are short and concise! Do not cite more than 5 words per sub-aspect!!!") -> str:
        """
        类似于 describe_topic_topwords_completion_object，但返回的是主题描述的字符串。
        调用 describe_topic_topwords_completion_object 并提取响应中的内容。
        Args:
            topwords (list[str]): List of topwords.
            n_words (int, optional): Number of words to use for the query. If None, all words are used.
            query_function (Callable, optional): Function to query the model. The function should take a list of topwords and return a string.

        Returns:
            str: A description of the topics by the model in the form of a string.
        """

        completion = self.describe_topic_topwords_completion_object(topwords, n_words, query_function)
        return completion.choices[0].message.content

    def generate_topic_name_str(self,
                                topwords: list[str],
                                n_words: int = None,
                                query_function: Callable = lambda
                                        tws: f"Please give me the common topic of those words: {tws}. Give me only the title of the topic and nothing else please. Make sure the title is precise and not longer than 5 words, ideally even shorter.") -> str:
        """
        生成一个主题名称。基于 topwords 和查询函数来生成主题名称。
        Args:
            topwords (list[str]): List of topwords.
            n_words (int, optional): Number of words to use for the query. If None, all words are used.
            query_function (Callable, optional): Function to query the model. The function should take a list of topwords and return a string.

        Returns:
            str: A topic name generated by the model in the form of a string.
        """

        return self.describe_topic_topwords_str(topwords, n_words, query_function)

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
            self.embedder.encoding_for_model(self.basic_model_instruction + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print(
                "Too many documents given. Using only the first part of the documents that fits into the context length. Number of documents used: ",
                np.argmax(tokens_cumsum > self.max_context_length))
            n_documents = np.argmax(tokens_cumsum > self.max_context_length)
            documents = documents[:n_documents]

        completion = self.client.chat.completions.create(model=self.openai_model,
                                                         messages=[
                                                             {"role": "system",
                                                              "content": self.basic_model_instruction + " " + self.corpus_instruction},
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
