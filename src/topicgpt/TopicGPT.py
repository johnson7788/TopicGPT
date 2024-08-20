import numpy as np
import os
from typing import Dict, Any
import pickle
# make sure the import works even if the package has not been installed and just the files are used
from topicgpt.Clustering import Clustering_and_DimRed
from topicgpt.ExtractTopWords import ExtractTopWords
from topicgpt.TopwordEnhancement import TopwordEnhancement
from topicgpt.GetEmbeddingsLocal import GetEmbeddingsLocal
from topicgpt.TopicPrompting import TopicPrompting
from topicgpt.TopicRepresentation import Topic
from topicgpt.Client import Client
import topicgpt.TopicRepresentation as TopicRepresentation


class TopicGPT:
    def __init__(self,
                 api_key: str = "",
                 base_url: str = "",
                 http_client=None,
                 n_topics: int = None,
                 openai_prompting_model: str = "gpt-3.5-turbo-16k",
                 max_number_of_tokens: int = 16384,
                 corpus_instruction: str = "",
                 document_embeddings: np.ndarray = None,
                 vocab_embeddings: dict[str, np.ndarray] = None,
                 embedding_client_url: str = None,
                 embedding_model: str = "text-embedding-ada-002",
                 max_number_of_tokens_embedding: int = 8191,
                 use_saved_embeddings: bool = True,
                 path_saved_embeddings: str = "",
                 clusterer: Clustering_and_DimRed = None,
                 n_topwords: int = 2000,
                 n_topwords_description: int = 500,
                 topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"],
                 compute_vocab_hyperparams: dict = {},
                 enhancer: TopwordEnhancement = None,
                 topic_prompting: TopicPrompting = None,
                 use_saved_topics: bool = True,  # 使用缓存
                 path_saved_topics: str = "",
                 documents:  list[dict] = [],
                 corpus: list[str] = [],
                 verbose: bool = True) -> None:

        """
        初始化TopicGPT
        Args:
            api_key (str)：你的 OpenAI API 密钥。请从 OpenAI API 密钥页面 获取此密钥。
            n_topics (int, optional)：要发现的主题数量。如果为 None，则使用 Hdbscan 算法（Hdbscan 项目页面）自动确定主题数量。否则，使用聚合聚类方法。请注意，如果数据不足，可能会发现比指定的主题数量更少的主题。
            openai_prompting_model (str, optional)：用于主题描述和提示的 OpenAI 模型。请参阅 OpenAI 模型文档 以了解可用的模型。
            max_number_of_tokens (int, optional)：用于 OpenAI API 的最大 token 数量。
            corpus_instruction (str, optional)：关于语料库的附加信息（如果有的话），以帮助模型更好地理解。
            document_embeddings (np.ndarray, optional)：语料库的文档嵌入。如果为 None，则将使用 OpenAI API 计算嵌入。
            vocab_embeddings (dict[str, np.ndarray], optional)：语料库的词汇嵌入，格式为字典，其中键为词汇，值为嵌入。如果为 None，则将使用 OpenAI API 计算嵌入。
            embedding_model (str, optional)：要使用的嵌入模型的名称。请参阅 OpenAI 嵌入模型文档 以了解可用的模型。
            max_number_of_tokens_embedding (int, optional)：计算嵌入时用于 OpenAI API 的最大 token 数量。
            use_saved_embeddings (bool, optional)：是否使用保存的嵌入。如果为 True，则从文件 SavedEmbeddings/embeddings.pkl 或其他 path_saved_embeddings 中加载嵌入。如果为 False，则使用 OpenAI API 计算嵌入并保存到文件中。
            path_saved_embeddings (str, optional)：保存的嵌入文件的路径。
            clusterer (Clustering_and_DimRed, optional)：聚类和降维对象。在 “Clustering/Clustering” 文件夹中查找该类。如果为 None，则使用具有默认参数的聚类对象。请注意，同时提供文档和词汇嵌入以及嵌入对象是没有意义的；clusterer 中指定的主题数量将覆盖 n_topics 参数。
            n_topwords (int, optional)：每个主题提取和保存的 top words 数量。请注意，后续可能会使用更少的 top words。
            n_topwords_description (int, optional)：提供给语言模型用于描述主题的 top words 数量。
            topword_extraction_methods (list[str], optional)：提取 top words 的方法列表。可用的方法包括 “tfidf”、“cosine_similarity” 和 “topword_enhancement”。有关更多细节，请参阅文件 ExtractTopWords/ExtractTopWords.py。
            compute_vocab_hyperparams (dict, optional)：计算词汇嵌入的超参数。有关更多细节，请参阅文件 ExtractTopWords/ExtractTopWords.py。
            enhancer (TopwordEnhancement, optional)：topword 增强对象。用于描述主题。在 “TopwordEnhancement/TopwordEnhancement.py” 文件夹中查找该类。如果为 None，则使用具有默认参数的 topword 增强对象。如果在此处指定了 OpenAI 模型，它将覆盖用于主题描述的 openai_prompting_model 参数。
            topic_prompting (TopicPrompting, optional)：用于制定提示的主题提示对象。在 “TopicPrompting/TopicPrompting.py” 文件夹中查找该类。如果为 None，则使用具有默认参数的主题提示对象。如果在此处指定了 OpenAI 模型，它将覆盖用于主题描述的 openai_prompting_model 参数。
            documents: 输入的原始的文档数据，里面包含很多meta信息
            corpus:  输入的原始的文本数据，文本数据列表
            verbose (bool, optional)：是否打印关于过程的详细信息。这可以通过传递的对象中的参数进行覆盖。
        """
        # 参数的检查
        assert api_key is not None, "您需要提供一个 OpenAI API 密钥。"
        assert n_topics is None or n_topics > 0, "主题数量需要是一个正整数。"
        assert max_number_of_tokens > 0, "最大 token 数量需要是一个正整数。"
        assert max_number_of_tokens_embedding > 0, "嵌入模型的最大 token 数量需要是一个正整数。"
        assert n_topwords > 0, "top words 的数量需要是一个正整数。"
        assert n_topwords_description > 0, "用于主题描述的 top words 数量需要是一个正整数。"
        assert len(
            topword_extraction_methods) > 0, "您需要提供至少一种 topword 提取方法。tfidf或者cosine_similarity或topword_enhancement"
        assert n_topwords_description <= n_topwords, "用于主题描述的 top words 数量需要小于或等于 top words 的数量。"
        self.api_key = api_key
        self.base_url = base_url
        self.client = Client(api_key=api_key, base_url=base_url, http_client=http_client)
        self.n_topics = n_topics
        self.openai_prompting_model = openai_prompting_model
        self.max_number_of_tokens = max_number_of_tokens
        self.corpus_instruction = corpus_instruction
        self.document_embeddings = document_embeddings
        self.vocab_embeddings = vocab_embeddings
        self.embedding_model = embedding_model
        self.max_number_of_tokens_embedding = max_number_of_tokens_embedding
        self.embedder = GetEmbeddingsLocal(client_url=embedding_client_url, embedding_model=embedding_model,
                                           max_tokens=self.max_number_of_tokens_embedding)
        self.clusterer = clusterer
        self.n_topwords = n_topwords
        self.n_topwords_description = n_topwords_description
        self.topword_extraction_methods = topword_extraction_methods
        self.compute_vocab_hyperparams = compute_vocab_hyperparams
        self.enhancer = enhancer
        self.topic_prompting = topic_prompting
        self.use_saved_embeddings = use_saved_embeddings
        self.use_saved_topics = use_saved_topics
        self.path_saved_embeddings = path_saved_embeddings
        self.path_saved_topics = path_saved_topics
        self.documents = documents # 占位用
        self.corpus = corpus  # 占位用
        self.verbose = verbose

        self.compute_vocab_hyperparams["verbose"] = self.verbose

        # if embeddings have already been downloaded to the folder SavedEmbeddings, then load them
        if self.use_saved_embeddings and os.path.exists(path_saved_embeddings):
            print(f"使用已保存的嵌入文件: {path_saved_embeddings}")
            with open(path_saved_embeddings, "rb") as f:
                self.document_embeddings, self.vocab_embeddings = pickle.load(f)

        for elem in topword_extraction_methods:
            assert elem in ["tfidf", "cosine_similarity",
                            "topword_enhancement"], "Invalid topword extraction method. Valid methods are 'tfidf', 'cosine_similarity', and 'topword_enhancement'."

        if clusterer is None:
            self.clusterer = Clustering_and_DimRed(number_clusters_hdbscan=self.n_topics, verbose=self.verbose)
        else:
            self.n_topics = clusterer.number_clusters_hdbscan

        if enhancer is None:
            self.enhancer = TopwordEnhancement(client=self.client, openai_model=self.openai_prompting_model,
                                               max_context_length=self.max_number_of_tokens,
                                               corpus_instruction=self.corpus_instruction, embedder=self.embedder)

        if topic_prompting is None:
            self.topic_prompting = TopicPrompting(topic_lis=[], client=self.client,
                                                  openai_prompting_model=self.openai_prompting_model,
                                                  max_context_length_promting=16000, enhancer=self.enhancer,
                                                  embedder=self.embedder,
                                                  max_context_length_embedding=self.max_number_of_tokens_embedding,
                                                  corpus_instruction=corpus_instruction)

        self.extractor = ExtractTopWords()

    def __repr__(self) -> str:
        repr = "TopicGPT object with the following parameters:\n"
        repr += "-" * 150 + "\n"
        repr += "n_topics: " + str(self.n_topics) + "\n"
        repr += "openai_prompting_model: " + self.openai_prompting_model + "\n"
        repr += "max_number_of_tokens: " + str(self.max_number_of_tokens) + "\n"
        repr += "corpus_instruction: " + self.corpus_instruction + "\n"
        repr += "embedding_model: " + self.embedding_model + "\n"
        repr += "clusterer: " + str(self.clusterer) + "\n"
        repr += "n_topwords: " + str(self.n_topwords) + "\n"
        repr += "n_topwords_description: " + str(self.n_topwords_description) + "\n"
        repr += "topword_extraction_methods: " + str(self.topword_extraction_methods) + "\n"
        repr += "compute_vocab_hyperparams: " + str(self.compute_vocab_hyperparams) + "\n"
        repr += "enhancer: " + str(self.enhancer) + "\n"
        repr += "topic_prompting: " + str(self.topic_prompting) + "\n"

        return repr

    def compute_embeddings(self, corpus: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        分别计算整个文档的嵌入和单词表的嵌入

        Args:
            corpus (list[str]): List of strings to embed, where each element is a document.

        Returns:
            tuple: A tuple containing two items:
                - document_embeddings (np.ndarray): Document embeddings for the corpus, with shape (len(corpus), n_embedding_dimensions).
                - vocab_embeddings (dict[str, np.ndarray]): Vocabulary embeddings for the corpus, provided as a dictionary where keys are words and values are embeddings.
        """

        self.document_embeddings = self.embedder.get_embeddings(corpus)["embeddings"]

        self.vocab_embeddings = self.extractor.embed_vocab_openAI(self.vocab, embedder=self.embedder)

        return self.document_embeddings, self.vocab_embeddings

    def extract_topics(self, corpus: list[str]) -> list[Topic]:
        """
        Extracts topics from the given corpus.

        Args:
            corpus (list[str]): List of strings to process, where each element represents a document.

        Returns:
            list[Topic]: A list of Topic objects representing the extracted topics.
        """

        assert self.document_embeddings is not None and self.vocab_embeddings is not None, "You need to compute the embeddings first."

        if self.vocab is None:
            self.vocab = self.extractor.compute_corpus_vocab(self.corpus, **self.compute_vocab_hyperparams)

        self.topic_lis = TopicRepresentation.extract_topics_no_new_vocab_computation(
            corpus=corpus,  # list[str]
            vocab=self.vocab,  # list[str]
            document_embeddings=self.document_embeddings,  # ndarry, [document_num, embedding_size]
            clusterer=self.clusterer,  # 聚类和降维度算法
            vocab_embeddings=self.vocab_embeddings,  # 每个词的嵌入
            n_topwords=self.n_topwords,  # 2000？
            topword_extraction_methods=self.topword_extraction_methods,  # ['tfidf', 'cosine_similarity']
            consider_outliers=True
        )

        return self.topic_lis

    def describe_topics(self, topics: list[Topic]) -> list[Topic]:
        """
        Names and describes the provided topics using the OpenAI API.

        Args:
            topics (list[Topic]): List of Topic objects to be named and described.

        Returns:
            list[Topic]: A list of Topic objects with names and descriptions.
        """

        assert self.topic_lis is not None, "You need to extract the topics first."

        if "cosine_similarity" in self.topword_extraction_methods:
            topword_method = "cosine_similarity"
        elif "tfidf" in self.topword_extraction_methods:
            topword_method = "tfidf"
        else:
            raise ValueError("You need to use either 'cosine_similarity' or 'tfidf' as topword extraction method.")

        self.topic_lis = TopicRepresentation.describe_and_name_topics(
            topics=topics,
            enhancer=self.enhancer,
            topword_method=topword_method,
            n_words=self.n_topwords_description
        )

        return self.topic_lis

    def load_cache_topics(self):
        """
        加载缓存的主题模型
        """
        if self.use_saved_topics and os.path.exists(self.path_saved_topics):
            with open(self.path_saved_topics, "rb") as f:
                self.topic_lis, self.vocab_embeddings, self.document_embeddings, self.vocab, self.corpus, self.topic_lis = pickle.load(
                    f)
            return True
        else:
            print(f"没有找到缓存的主题模型，请先训练主题模型。{self.path_saved_topics}")
            return False

    def fit(self, documents: list[dict], verbose: bool = True):
        """
        # 流程
        计算单词表： compute_corpus_vocab，停用词、词频和文档频率初始化：使用 jieba.lcut 分词，将每个文档切分成词语列表。词频计算：指定的频率筛选，词汇表排序和返回：
        嵌入计算： 分别计算整个文档的嵌入和单词表的嵌入
        主题提取： 使用extract_topics_no_new_vocab_computation，降维和聚类，提取中心点，每个文档降维，然后计算文档和质心之间的相似性
        主题描述： 使用describe_topic_words，给聚类的名称总结1个名字，然后写一些描述
        Args:
            documents (list[dict]): 所有文档预料
            verbose (bool, optional):  True or False. Defaults to True.
        """
        self.documents = documents
        self.corpus = [document['content'] for document in self.documents]
        # remove empty documents
        if self.use_saved_topics and os.path.exists(self.path_saved_topics):
            self.load_cache_topics()
        else:
            print(f"不使用缓存，重新计算主题")
            if self.vocab_embeddings is None:
                if verbose:
                    print("Computing vocabulary...")
                self.vocab = self.extractor.compute_corpus_vocab(self.corpus, **self.compute_vocab_hyperparams)
            else:
                print('Vocab already computed')
                self.vocab = list(self.vocab_embeddings.keys())

            if self.vocab_embeddings is None or self.document_embeddings is None:
                if verbose:
                    print("计算单词表向量和文档向量")
                self.compute_embeddings(corpus=self.corpus)
            else:
                print('Embeddings already computed')
            if verbose:
                print("Extracting topics...")
            self.topic_lis = self.extract_topics(corpus=self.corpus)
            # self.topic_lis: [Topic: 0, Topic: 1]
            if verbose:
                print("使用LLM解释聚类后生成的主题")
            self.topic_lis = self.describe_topics(topics=self.topic_lis)
        self.topic_prompting.topic_lis = self.topic_lis
        self.topic_prompting.vocab_embeddings = self.vocab_embeddings
        self.topic_prompting.vocab = self.vocab

    def visualize_clusters_prepare_data(self):
        """
        该函数用于可视化已识别的聚类，展示主题的散点图。
            确保已提取主题。
            合并所有主题的文档嵌入（document_embeddings_hd）。
            合并所有文档文本。
            生成文档的索引。
            获取每个主题的名称。
            调用 visualize_clusters_dynamic 函数绘制动态聚类可视化。
        """
        # 确保已提取主题
        assert self.topic_lis is not None, "请先提取主题"
        # 合并所有主题的文档嵌入（document_embeddings_hd）。 形状[document_nums, hidden_size)
        all_document_embeddings = np.concatenate([topic.document_embeddings_hd for topic in self.topic_lis], axis=0)
        # 合并所有文档文本。 list[str]
        all_texts = np.concatenate([topic.documents for topic in self.topic_lis], axis=0)
        # 生成文档的索引。主题的类别序号，eg: [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
        all_document_indices = np.concatenate(
            [np.repeat(i, topic.document_embeddings_hd.shape[0]) for i, topic in enumerate(self.topic_lis)], axis=0)
        # 获取每个主题的名称。list[str], eg: ['Topic 0: \n"Convenient meal delivery service"\n', 'Topic 1: \nTitle: Online Food Ordering Convenience\n']
        class_names = [str(topic) for topic in self.topic_lis]
        # 降维到2D
        embeddings_2d = self.clusterer.visualize_2D_data_prepare(embeddings=all_document_embeddings)
        # 调用 visualize_clusters_dynamic 函数绘制动态聚类可视化。
        return all_document_embeddings, all_document_indices, all_texts, class_names, embeddings_2d

    def visualize_clusters(self):
        """
        该函数用于可视化已识别的聚类，展示主题的散点图。
            确保已提取主题。
            合并所有主题的文档嵌入（document_embeddings_hd）。
            合并所有文档文本。
            生成文档的索引。
            获取每个主题的名称。
            调用 visualize_clusters_dynamic 函数绘制动态聚类可视化。
        """
        # 确保已提取主题
        all_document_embeddings, all_document_indices, all_texts, class_names = self.visualize_clusters_prepare_data()
        # 调用 visualize_clusters_dynamic 函数绘制动态聚类可视化。
        self.clusterer.visualize_clusters_dynamic(all_document_embeddings, all_document_indices, all_texts, class_names)
        return True

    def repr_topics(self) -> str:
        """
        返回所有主题的摘要。
        """

        assert self.topic_lis is not None, "You need to extract the topics first."

        if "cosine_similarity" in self.topword_extraction_methods:
            topword_method = "cosine_similarity"
        elif "tfidf" in self.topword_extraction_methods:
            topword_method = "tfidf"
        else:
            raise ValueError("You need to use either 'cosine_similarity' or 'tfidf' as topword extraction method.")

        repr = ""
        for topic in self.topic_lis:
            repr += str(topic) + "\n"
            repr += "Topic_description: " + topic.topic_description + "\n"
            repr += "Top words: " + str(topic.top_words[topword_method][:10]) + "\n"
            repr += "\n"
            repr += "-" * 150 + "\n"

        return repr

    def print_topics(self):
        """
        Prints a string explanation of the topics.
        """
        print(self.repr_topics())

    def prompt(self, query: str) -> tuple[str, object]:
        """
        Prompts the model with the given query.

        Args:
            query (str): The query to prompt the model with.

        Returns:
            tuple: A tuple containing two items:
                - answer (str): The answer from the model.
                - function_result (object): The result of the function call.
        
        Note:
            Please refer to the TopicPrompting class for more details on available functions for prompting the model.
        """

        result = self.topic_prompting.general_prompt(query)

        answer = result[0][-1].choices[0].message.content
        function_result = result[1]
        self.topic_prompting._fix_dictionary_topwords()
        self.topic_lis = self.topic_prompting.topic_lis

        return answer, function_result

    def pprompt(self, query: str, return_function_result: bool = True) -> object:
        """
        Prompts the model with the given query and prints the answer.

        Args:
            query (str): The query to prompt the model with.
            return_function_result (bool, optional): Whether to return the result of the function call by the Language Model (LLM).

        Returns:
            object: The result of the function call if return_function_result is True, otherwise None.
        """

        answer, function_result = self.prompt(query)

        print(answer)

        if return_function_result:
            return function_result

    def save_embeddings(self, path: str = None) -> None:
        """
        Saves the document and vocabulary embeddings to a pickle file for later re-use.

        Args:
            path (str, optional): The path to save the embeddings to. Defaults to embeddings_path.
        """
        if path is None:
            path = self.path_saved_embeddings
        print(f"保存embedding到{path}")
        assert self.document_embeddings is not None and self.vocab_embeddings is not None, "You need to compute the embeddings first."

        # create dictionary if it doesn't exist yet 
        if not os.path.exists("SavedEmbeddings"):
            os.makedirs("SavedEmbeddings")
        with open(path, "wb") as f:
            pickle.dump([self.document_embeddings, self.vocab_embeddings], f)
        print(f"保存embedding到{path}成功")

    def save_topics(self, path: str = None) -> None:
        """
        保存主题相关到本地

        Args:
            path (str, optional): The path to save the embeddings to. Defaults to embeddings_path.
        """
        if path is None:
            path = self.path_saved_topics
        print(f"保存主题到{path}")
        assert self.topic_lis is not None and self.vocab_embeddings is not None, "你应该先计算好主题，然后保存主题"
        assert self.document_embeddings is not None and self.vocab is not None, "你应该先计算好主题，然后保存主题"

        # create dictionary if it doesn't exist yet
        if not os.path.exists("SavedEmbeddings"):
            os.makedirs("SavedEmbeddings")

        with open(path, "wb") as f:
            pickle.dump([self.topic_lis, self.vocab_embeddings, self.document_embeddings, self.vocab, self.corpus,
                         self.topic_lis], f)
        print(f"保存主题到{path}成功")

    def to_dict(self) -> Dict[str, Any]:
        # 转换为字典（需要排除无法序列化的对象）
        return {
            'api_key': self.api_key,
            'base_url': self.base_url,
            'n_topics': self.n_topics,
            'openai_prompting_model': self.openai_prompting_model,
            'max_number_of_tokens': self.max_number_of_tokens,
            'corpus_instruction': self.corpus_instruction,
            'document_embeddings': self.document_embeddings.tolist() if self.document_embeddings is not None else None,
            'vocab_embeddings': {k: v.tolist() for k, v in
                                 self.vocab_embeddings.items()} if self.vocab_embeddings is not None else None,
            'embedding_model': self.embedding_model,
            'max_number_of_tokens_embedding': self.max_number_of_tokens_embedding,
            'use_saved_embeddings': self.use_saved_embeddings,
            'path_saved_embeddings': self.path_saved_embeddings,
            'clusterer': str(self.clusterer),  # 或根据需要自定义序列化方式
            'n_topwords': self.n_topwords,
            'n_topwords_description': self.n_topwords_description,
            'topword_extraction_methods': self.topword_extraction_methods,
            'compute_vocab_hyperparams': self.compute_vocab_hyperparams,
            'enhancer': str(self.enhancer),  # 或根据需要自定义序列化方式
            'topic_prompting': str(self.topic_prompting),  # 或根据需要自定义序列化方式
            'use_saved_topics': self.use_saved_topics,
            'path_saved_topics': self.path_saved_topics,
            'documents': self.documents,
            'corpus': self.corpus,
            'verbose': self.verbose,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        # 从字典创建实例
        return cls(
            api_key=data.get('api_key', ""),
            base_url=data.get('base_url', ""),
            n_topics=data.get('n_topics'),
            openai_prompting_model=data.get('openai_prompting_model', "gpt-3.5-turbo-16k"),
            max_number_of_tokens=data.get('max_number_of_tokens', 16384),
            corpus_instruction=data.get('corpus_instruction', ""),
            document_embeddings=np.array(data.get('document_embeddings')) if data.get(
                'document_embeddings') is not None else None,
            vocab_embeddings={k: np.array(v) for k, v in data.get('vocab_embeddings', {}).items()} if data.get(
                'vocab_embeddings') is not None else None,
            embedding_model=data.get('embedding_model', "text-embedding-ada-002"),
            max_number_of_tokens_embedding=data.get('max_number_of_tokens_embedding', 8191),
            use_saved_embeddings=data.get('use_saved_embeddings', True),
            path_saved_embeddings=data.get('path_saved_embeddings', ""),
            clusterer=None,  # 自定义加载方式
            n_topwords=data.get('n_topwords', 2000),
            n_topwords_description=data.get('n_topwords_description', 500),
            topword_extraction_methods=data.get('topword_extraction_methods', ["tfidf", "cosine_similarity"]),
            compute_vocab_hyperparams=data.get('compute_vocab_hyperparams', {}),
            enhancer=None,  # 自定义加载方式
            topic_prompting=None,  # 自定义加载方式
            use_saved_topics=data.get('use_saved_topics', True),
            path_saved_topics=data.get('path_saved_topics', ""),
            documents = data.get('documents', []),
            corpus =  data.get('corpus', []),
            verbose=data.get('verbose', True),
        )
