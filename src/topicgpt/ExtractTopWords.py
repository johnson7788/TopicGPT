import nltk
import string
import collections
from tqdm import tqdm
from typing import List
import numpy as np
import re
import jieba
import umap
from collections import Counter
import warnings
from typing import List

# make sure the import works even if the package has not been installed and just the files are used
try:
    from topicgpt.GetEmbeddingsLocal import GetEmbeddingsLocal
except:
    from GetEmbeddingsLocal import GetEmbeddingsLocal

class ExtractTopWords:
    # 用于从文本数据中提取和分析高频词汇和主题词汇。
    def extract_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """
        Extract centroids of clusters.
        提取每个聚类的质心（即每个聚类中心点）。
        输入是嵌入矩阵 embeddings 和聚类标签 labels。
        返回值是一个字典，键是聚类标签，值是相应质心。
        Args:
            embeddings (np.ndarray): Embeddings to cluster and reduce.
            labels (np.ndarray): Cluster labels. -1 means outlier.

        Returns:
            dict: Dictionary of cluster labels and their centroids.
        """

        centroid_dict = {}
        for label in np.unique(labels):
            if label != -1:
                # 离群点不应该有质心
                centroid_dict[label] = np.mean(embeddings[labels == label], axis = 0)

        return centroid_dict
    
    def extract_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Extract the single centroid of a cluster.
        提取单个聚类的质心。
        输入是嵌入矩阵 embeddings。
        返回值是质心。
        Args:
            embeddings (np.ndarray): Embeddings to extract the centroid from.

        Returns:
            np.ndarray: The centroid of the cluster.
        """

        return np.mean(embeddings, axis = 0)
    
    def compute_centroid_similarity(self, embeddings: np.ndarray, centroid_dict: dict, cluster_label: int) -> np.ndarray:
        """
        计算文档嵌入与聚类质心的余弦相似度。
        输入是嵌入矩阵 embeddings、质心字典 centroid_dict 和聚类标签 cluster_label。
        返回值是相似度数组。
        Args:
            embeddings (np.ndarray): Embeddings to cluster and reduce.
            centroid_dict (dict): Dictionary of cluster labels and their centroids.
            cluster_label (int): Cluster label for which to compute the similarity.

        Returns:
            np.ndarray: Cosine similarity of the document embeddings to the centroid of the cluster.
        """

        centroid = centroid_dict[cluster_label]
        similarity = np.dot(embeddings, centroid) / (np.linalg.norm(embeddings) * np.linalg.norm(centroid))
        return similarity
    
    def get_most_similar_docs(self, corpus: list[str], embeddings: np.ndarray, labels: np.ndarray, centroid_dict: dict, cluster_label: int, top_n: int = 10) -> List[str]:
        """
        获取与聚类质心最相似的文档。
        输入是文档列表 corpus、嵌入矩阵 embeddings、聚类标签 labels、质心字典 centroid_dict、聚类标签 cluster_label 和提取的文档数量 top_n。
        返回值是最相似的文档列表。
        Args:
            corpus (list[str]): List of documents.
            embeddings (np.ndarray): Embeddings to cluster and reduce.
            labels (np.ndarray): Cluster labels. -1 means outlier.
            centroid_dict (dict): Dictionary of cluster labels and their centroids.
            cluster_label (int): Cluster label for which to compute the similarity.
            top_n (int, optional): Number of top documents to extract.

        Returns:
            List[str]: List of the most similar documents to the centroid of a cluster.
        """

        similarity = self.compute_centroid_similarity(embeddings, centroid_dict, cluster_label)
        most_similar_docs = [corpus[i] for i in np.argsort(similarity)[-top_n:][::-1]]
        return most_similar_docs
    
    def compute_corpus_vocab(self, 
                        corpus: list[str],
                        remove_stopwords: bool = True, 
                        remove_punction: bool = True, 
                        min_word_length: int = 2,
                        max_word_length: int = 20, 
                        remove_short_words: bool = True, 
                        remove_numbers: bool = True, 
                        verbose: bool = True,
                        min_doc_frequency: int = 3,
                        min_freq: float = 0.1,
                        max_freq: float = 0.9) -> list[str]:
        """
        计算语料库的词汇表，并对语料库进行预处理。
        输入包括文档列表 corpus 和一系列可选参数。
        返回值是词汇表的排序列表。
        停用词、词频和文档频率初始化：使用 jieba.lcut 分词，将每个文档切分成词语列表。词频计算：指定的频率筛选，词汇表排序和返回：
        Args:
            corpus: list[str]：文档列表，即语料库。
            remove_stopwords: bool：是否移除停用词（如“的”、“是”等高频无意义的词）。
            remove_punction: bool：是否移除标点符号。
            min_word_length: int：保留的最小词长。
            max_word_length: int：保留的最大词长。
            remove_short_words: bool：是否移除长度小于 min_word_length 的词。
            remove_numbers: bool：是否移除包含数字的词。
            verbose: bool：是否输出进度信息及其他描述信息。
            min_doc_frequency: int：词语在多少文档中出现才会被纳入词汇表。
            min_freq: float：词语在语料库中出现频率的最小百分位数。
            max_freq: float：词语在语料库中出现频率的最大百分位数。

        Returns:
            list[str]: List of words in the corpus sorted alphabetically.
        """

        stopwords = set(nltk.corpus.stopwords.words('chinese'))
        
        word_counter = collections.Counter()
        doc_frequency = collections.defaultdict(set)

        for doc_id, doc in enumerate(tqdm(corpus, disable=not verbose, desc="Processing corpus")):
            words = jieba.lcut(doc)
            for word in words:
                if remove_punction and word in string.punctuation:
                    continue
                if remove_stopwords and word.lower() in stopwords:
                    continue
                if remove_numbers and re.search(r'\d', word):  # use a regular expression to check for digits
                    continue
                # remove words that do not begin with an alphabetic character
                if not word[0].isalpha():
                    continue
                if len(word) > max_word_length or (remove_short_words and len(word) < min_word_length):
                    continue
                
                word_lower = word.lower()
                word_counter[word_lower] += 1
                doc_frequency[word_lower].add(doc_id)

        total_words = sum(word_counter.values())
        freq_counter = {word: count / total_words for word, count in word_counter.items()}

        # print most common words and their frequencies
        if verbose:
            print("Most common words in the vocabulary:")
            for word, count in word_counter.most_common(10):
                print(f"{word}: {count}")

        freq_arr = np.array(list(freq_counter.values()))

        min_freq_value = np.quantile(freq_arr, min_freq, method="lower")
        max_freq_value = np.quantile(freq_arr, max_freq, method="higher")
        

        vocab = {}

        for word in freq_counter.keys():
            if min_freq_value <= freq_counter[word] <= max_freq_value and len(doc_frequency[word]) >= min_doc_frequency:
                vocab[word] = freq_counter[word]

        vocab = {word for word in freq_counter.keys() 
                if min_freq_value <= freq_counter[word] <= max_freq_value 
                and len(doc_frequency[word]) >= min_doc_frequency}

        # Sorting the vocabulary alphabetically
        vocab = sorted(list(vocab))
        
        return vocab

    def compute_words_topics(self, corpus: list[str], vocab: list[str], labels: np.ndarray) -> dict:
        """
        计算每个主题的词汇。
        输入是文档列表 corpus、词汇表 vocab 和聚类标签 labels。
        返回值是主题词汇的字典。
        Args:
            corpus (list[str]): List of documents.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 means outlier.

        Returns:
            dict: Dictionary of topics and their words.
        """
        # Download NLTK resources (only required once)
        nltk.download("punkt")
        vocab = set(vocab)

        words_per_topic = {label: [] for label in np.unique(labels) if label != -1}

        for doc, label in tqdm(zip(corpus, labels), desc="Computing words per topic", total=len(corpus)):
            if label != -1:
                words = jieba.lcut(doc)
                for word in words:
                    if word.lower() in vocab:
                        words_per_topic[label].append(word.lower())

        return words_per_topic
                    
    def embed_vocab_openAI(self, vocab: list[str], embedder: GetEmbeddingsLocal = None) -> dict[str, np.ndarray]:
        """
        获取单词表中每个词的嵌入
        输入是客户端 client、词汇表 vocab 和可选的嵌入对象 embedder。
        返回值是词汇和其嵌入的字典。
        Args:
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            embedder (GetEmbeddingsLocal, optional): Embedding object.

        Returns:
            dict[str, np.ndarray]: Dictionary of words and their embeddings.
        """

        vocab = sorted(list(set(vocab)))
        result = embedder.get_embeddings(vocab)

        res_dict = {}
        for word, emb in zip(vocab, result["embeddings"]):
            res_dict[word] = emb
        return res_dict
    
    def compute_bow_representation(self, document: str, vocab: list[str], vocab_set: set[str]) -> np.ndarray:
        """
        计算文档的词袋表示。
        输入是文档 document、词汇表 vocab 和词汇集合 vocab_set。
        返回值是词袋表示的数组。

        Args:
            document (str): Document to compute the bag-of-words representation of.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            vocab_set (set[str]): Set of words in the corpus sorted alphabetically.

        Returns:
            np.ndarray: Bag-of-words representation of the document.
        """

        bow = np.zeros(len(vocab))
        words = jieba.lcut(document)
        if vocab_set is None:
            vocab_set = set(vocab)
        for word in words:
            if word.lower() in vocab_set:
                bow[vocab.index(word.lower())] += 1
        return bow   

    def compute_word_topic_mat(self, corpus: list[str], vocab: list[str], labels: np.ndarray, consider_outliers=False) -> np.ndarray:
        """
        高效地计算词-主题矩阵。
        输入是文档列表 corpus、词汇表 vocab 和聚类标签 labels。
        返回值是词-主题矩阵。

        Args:
            corpus (list[str]): List of documents.
            vocab (list[str]): List of words in the corpus, sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 indicates outliers.
            consider_outliers (bool, optional): Whether to consider outliers when computing the top words. Defaults to False.

        Returns:
            np.ndarray: Word-topic matrix.
        """

        # 变成numpy array数组
        corpus_arr = np.array(corpus) 
        # consider_outliers考虑不考虑有啥区别， word_topic_mat的格式[vocab_size, num_topics],统计每个词在每个主题下的出现次数
        if consider_outliers:
            word_topic_mat = np.zeros((len(vocab), len((np.unique(labels)))))
        else:
            word_topic_mat = np.zeros((len(vocab), len((np.unique(labels)))))
        
        for i, label in tqdm(enumerate(np.unique(labels)), desc="计算词主题矩阵", total=len(np.unique(labels))):
            topic_docs = corpus_arr[labels == label]  #对应聚类标签label的语句
            topic_doc_string = " ".join(topic_docs)
            topic_doc_words = jieba.lcut(topic_doc_string)  #重新分词？？
            topic_doc_counter = Counter(topic_doc_words)  #计算词频

            word_topic_mat[:, i] = np.array([topic_doc_counter.get(word, 0) for word in vocab])
        #词典中的每个词
        return word_topic_mat

    def compute_word_topic_mat_words(self, words: list[str], vocab: list[str], labels: np.ndarray) -> np.ndarray:
        """
        高效地计算词-主题矩阵，不用于主题聚类相关，只用于词的聚类
        输入是words 和聚类标签 labels。
        返回值是词-主题矩阵。需要去掉离群点，否则影响tfidf和cosine的计算

        Args:
            words (list[str]): List of words in the corpus.
            vocab (list[str]): 单词表，去重后的所有词。
            labels (np.ndarray): Cluster labels. -1 indicates outliers.
            consider_outliers (bool, optional): Whether to consider outliers when computing the top words. Defaults to False.

        Returns:
            np.ndarray: Word-topic matrix.
        """
        # 获取有效的标签（如果不考虑outliers，则移除-1的标签）
        unique_labels = np.unique(labels[labels != -1])
        # 对label进行排序，确保顺序是由小到大，因为位置就是对应这标签了, 标签是从0开始
        unique_labels = np.sort(unique_labels)

        # 初始化矩阵，行数为词汇表的大小，列数为有效标签的数量
        word_topic_mat = np.zeros((len(vocab), len(unique_labels)))

        # 将词汇映射到其在词汇表中的索引
        word_to_index = {word: i for i, word in enumerate(vocab)}

        for i, label in tqdm(enumerate(unique_labels), desc="计算词主题频率矩阵", total=len(unique_labels)):
            # 获取属于当前聚类的词
            cluster_words = [words[idx] for idx, l in enumerate(labels) if l == label]
            # 计算词频并更新矩阵
            for word in cluster_words:
                if word in word_to_index:
                    word_topic_mat[word_to_index[word], i] += 1
        #包含异常点
        return word_topic_mat

    def extract_topwords_tfidf(self, word_topic_mat: np.ndarray, vocab: list[str], labels: np.ndarray, top_n_words: int = 10) -> dict:
        """
        使用基于类的tf-idf得分提取每个主题的高频词。
        输入是词-主题矩阵 word_topic_mat、词汇表 vocab 和聚类标签 labels。
        返回值是每个主题的高频词字典。
        Args:
            word_topic_mat (np.ndarray): Word-topic matrix.
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            labels (np.ndarray): Cluster labels. -1 means outlier.
            top_n_words (int, optional): Number of top words to extract per topic.

        Returns:
            dict: Dictionary of topics and their top words.
        """

        #
        # if min(labels) == -1:
        #     word_topic_mat = word_topic_mat[:, 1:]  # 默认最后一个维度是-1异常点的维度

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            tf = word_topic_mat / np.sum(word_topic_mat, axis=0)
            idf = np.log(1 + (word_topic_mat.shape[1] / np.sum(word_topic_mat > 0, axis=1)))

            tfidf = tf * idf[:, np.newaxis]
        
            # set tfidf to zero if tf is nan (happens if word does not occur in any document or topic does not have any words)
            tfidf[np.isnan(tf)] = 0

        # extract top words for each topic
        top_words = {}
        top_word_scores = {}
        for topic in np.unique(labels):
            if topic != -1:
                indices = np.argsort(-tfidf[:, topic])[:top_n_words]
                top_words[topic] = [vocab[word_idx] for word_idx in indices]
                top_word_scores[topic] = [tfidf[word_idx, topic] for word_idx in indices]

        # {0:[vocabs],1:[vocabs]}, top_word_scores: {0:[scores],1:[scores]}, top_words长度是vocabs的长度, scores长度是vocabs的长度
        return top_words, top_word_scores

    def extract_topwords_tfidf_words(self, word_topic_mat: np.ndarray, vocab: list[str], top_n_words: int = 10) -> dict:
        """
        使用基于类的tf-idf得分提取每个类别的高频词。
        输入是词-类别矩阵 word_topic_mat 和词汇表 vocab。
        返回值是每个类别的高频词字典。

        Args:
            word_topic_mat (np.ndarray): Word-topic (category) matrix.
            vocab (list[str]): List of words in the corpus.
            top_n_words (int, optional): Number of top words to extract per topic.

        Returns:
            dict: Dictionary of topics and their top words.
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # 计算TF (term frequency)
            tf = word_topic_mat / np.sum(word_topic_mat, axis=0)

            # 计算IDF (inverse document frequency)
            idf = np.log(1 + (word_topic_mat.shape[1] / np.sum(word_topic_mat > 0, axis=1)))

            # 计算TF-IDF
            tfidf = tf * idf[:, np.newaxis]

            # 如果TF为NaN，设为0
            tfidf[np.isnan(tf)] = 0

        # 提取每个类别的Top N高频词
        top_words = {}
        top_word_scores = {}

        for topic in range(word_topic_mat.shape[1]):
            # 获取排序结果
            sorted_indices = np.argsort(-tfidf[:, topic])

            # 过滤掉tfidf为0的结果
            filtered_indices = [idx for idx in sorted_indices if tfidf[idx, topic] > 0]

            # 只获取Top N的词汇
            top_indices = filtered_indices[:top_n_words]
            # 获取对应的词语
            top_words[topic] = [vocab[word_idx] for word_idx in top_indices]
            top_word_scores[topic] = [tfidf[word_idx, topic] for word_idx in top_indices]

        return top_words, top_word_scores

    def compute_embedding_similarity_centroids(self, vocab: list[str], vocab_embedding_dict: dict, umap_mapper: umap.UMAP, centroid_dict: dict, reduce_vocab_embeddings: bool = False, reduce_centroid_embeddings: bool = False) -> np.ndarray:
        """
        计算词汇到聚类质心的余弦相似度。
        输入是词汇表 vocab、词汇嵌入字典 vocab_embedding_dict、UMAP映射器 umap_mapper 和质心字典 centroid_dict。
        返回值是词汇到质心的余弦相似度矩阵。
        Args:
            vocab (list[str]): List of words in the corpus sorted alphabetically.
            vocab_embedding_dict (dict): Dictionary of words and their embeddings.
            umap_mapper (umap.UMAP): UMAP mapper to transform new embeddings in the same way as the document embeddings.
            centroid_dict (dict): Dictionary of cluster labels and their centroids. -1 means outlier.
            reduce_vocab_embeddings (bool, optional): Whether to reduce the vocab embeddings with the UMAP mapper.
            reduce_centroid_embeddings (bool, optional): Whether to reduce the centroid embeddings with the UMAP mapper.

        Returns:
            np.ndarray: Cosine similarity of each word in the vocab to each centroid. Has shape (len(vocab), len(centroid_dict) - 1).
        """

        embedding_dim = umap_mapper.n_components
        centroid_arr = np.zeros((len(centroid_dict), embedding_dim))
        for i, centroid in enumerate(centroid_dict.values()):
            centroid_arr[i] = centroid
        if reduce_centroid_embeddings:
            centroid_arr = umap_mapper.transform(centroid_arr)
        # 计算质心，[3,5]
        centroid_arr = centroid_arr / np.linalg.norm(centroid_arr, axis=1).reshape(-1,1)
        
        # eg: org_embedding_dim: 768
        org_embedding_dim = list(vocab_embedding_dict.values())[0].shape[0]
        vocab_arr = np.zeros((len(vocab), org_embedding_dim))
        for i, word in enumerate(vocab):
            vocab_arr[i] = vocab_embedding_dict[word]
        if reduce_vocab_embeddings:
            vocab_arr = umap_mapper.transform(vocab_arr)
        #词典中每个词的嵌入降维后的质心, [vocab_size, 5]
        vocab_arr = vocab_arr / np.linalg.norm(vocab_arr, axis=1).reshape(-1,1)
        #similarity: [29,3]
        similarity = vocab_arr @ centroid_arr.T # cosine similarity
        return similarity
    
    def extract_topwords_centroid_similarity(self, word_topic_mat: np.ndarray, vocab: list[str], vocab_embedding_dict: dict, centroid_dict: dict, umap_mapper: umap.UMAP, top_n_words: int = 10, reduce_vocab_embeddings: bool = True, reduce_centroid_embeddings: bool = False, consider_outliers: bool = False) -> tuple[dict, np.ndarray]:
        """
        通过计算词汇到聚类质心的余弦相似度提取每个聚类的高频词。
        输入是词-主题矩阵 word_topic_mat、词汇表 vocab、词汇嵌入字典 vocab_embedding_dict、质心字典 centroid_dict、UMAP映射器 umap_mapper 和提取的高频词数量 top_n_words。
        返回值是高频词字典和词汇到质心的余弦相似度矩阵。
        Args:
            word_topic_mat (np.ndarray): Word-topic matrix.形状[vocab_size, num_topics]
            vocab (list[str]): 单词表
            vocab_embedding_dict (dict): 单词表对应的嵌入
            centroid_dict (dict):标签对应的质心的字典 -1 means outlier.
            umap_mapper (umap.UMAP): UMAP mapper to transform new embeddings in the same way as the document embeddings.
            top_n_words (int, optional): Number of top words to extract per topic.
            reduce_vocab_embeddings (bool, optional): Whether to reduce the vocab embeddings with the UMAP mapper.
            reduce_centroid_embeddings (bool, optional): Whether to reduce the centroid embeddings with the UMAP mapper.
            consider_outliers (bool, optional): Whether to consider outliers when computing the top words. I.e., whether the labels contain -1 to indicate outliers.

        Returns:
            dict: Dictionary of topics and their top words.
            np.ndarray: Cosine similarity of each word in the vocab to each centroid. Has shape (len(vocab), len(centroid_dict) - 1).
        """
        #[vocab_size, 3]，计算每个词和质心的相似度
        similarity_mat = self.compute_embedding_similarity_centroids(vocab, vocab_embedding_dict, umap_mapper, centroid_dict, reduce_vocab_embeddings, reduce_centroid_embeddings)
        top_words = {}
        top_word_scores = {}

        assert similarity_mat.shape == word_topic_mat.shape, "每个词和质心相似性矩阵的形状应该和每个词的主题矩阵的形状相同"
        for i, topic in enumerate(np.unique(list(centroid_dict.keys()))):
            topic_similarity_mat = similarity_mat[:, topic] * word_topic_mat[:, topic]
            # 同样的道理，过滤掉相似性为0的，防止有的时候，单词的数量比较少，但是top_n_words比较大，那么相似为0的也会放到最后了
            # 获取排序结果
            sorted_indices = np.argsort(-topic_similarity_mat)

            # 过滤掉相似性为0的结果
            filtered_indices = [idx for idx in sorted_indices if topic_similarity_mat[idx] > 0]

            # 只获取Top N的词汇
            top_indices = filtered_indices[:top_n_words]
            top_words[topic] = [vocab[word_idx] for word_idx in top_indices]
            top_word_scores[topic] = [topic_similarity_mat[word_idx] for word_idx in top_indices]
        # {0: ['word1', 'word2', ...], 1: ['word3', 'word4', ...], ...}, # 单词标签对应
        return top_words, top_word_scores