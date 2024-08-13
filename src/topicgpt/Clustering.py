import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import umap.plot
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple


class Clustering_and_DimRed():
    """
    用于执行UMAP降维和HDBSCAN聚类。它包括数据预处理、降维、聚类和可视化等功能。
    使用 UMAP (Uniform Manifold Approximation and Projection) 进行降维，并使用 HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) 进行聚类。
    UMAP 是一种非线性降维算法，能够将高维数据映射到低维空间中，而 HDBSCAN 是一种基于密度聚类的算法，能够将数据点自动分组。
    """

    def __init__(self,
                 n_dims_umap: int = 5,
                 n_neighbors_umap: int = 15,
                 min_dist_umap: float = 0,
                 metric_umap: str = "cosine",
                 min_cluster_size_hdbscan: int = 30,
                 metric_hdbscan: str = "euclidean",
                 cluster_selection_method_hdbscan: str = "eom",
                 number_clusters_hdbscan: int = None,
                 random_state: int = 42,
                 verbose: bool = True,
                 UMAP_hyperparams: dict = {},
                 HDBSCAN_hyperparams: dict = {}) -> None:
        """
        Initializes the clustering and dimensionality reduction parameters for topic modeling.

        Args:
            n_dims_umap (int, optional): Number of dimensions to reduce to using UMAP.
            n_neighbors_umap (int, optional): Number of neighbors for UMAP.
            min_dist_umap (float, optional): Minimum distance for UMAP.
            metric_umap (str, optional): Metric for UMAP.
            min_cluster_size_hdbscan (int, optional): Minimum cluster size for HDBSCAN.
            metric_hdbscan (str, optional): Metric for HDBSCAN.
            cluster_selection_method_hdbscan (str, optional): Cluster selection method for HDBSCAN.
            number_clusters_hdbscan (int, optional): Number of clusters for HDBSCAN. If None, HDBSCAN will determine the number of clusters automatically. Ensure that min_cluster_size is not too large to find enough clusters.
            random_state (int, optional): Random state for UMAP and HDBSCAN.
            verbose (bool, optional): Whether to print progress.
            UMAP_hyperparams (dict, optional): Additional hyperparameters for UMAP.
            HDBSCAN_hyperparams (dict, optional): Additional hyperparameters for HDBSCAN.
        """

        # do some checks on the input arguments
        assert n_dims_umap > 0, "n_dims_umap must be greater than 0"
        assert n_neighbors_umap > 0, "n_neighbors_umap must be greater than 0"
        assert min_dist_umap >= 0, "min_dist_umap must be greater than or equal to 0"
        assert min_cluster_size_hdbscan > 0, "min_cluster_size_hdbscan must be greater than 0"
        assert number_clusters_hdbscan is None or number_clusters_hdbscan > 0, "number_clusters_hdbscan must be greater than 0 or None"
        assert random_state is None or random_state >= 0, "random_state must be greater than or equal to 0"
        if isinstance(number_clusters_hdbscan, int) and number_clusters_hdbscan < min_cluster_size_hdbscan:
            # 类别的最小数量必须大于类别数量
            min_cluster_size_hdbscan = int(number_clusters_hdbscan / 2)
            if min_cluster_size_hdbscan == 1:  # 最小聚类数量不能为1
                min_cluster_size_hdbscan = 2
        self.random_state = random_state
        self.verbose = verbose
        self.UMAP_hyperparams = UMAP_hyperparams  # {'n_components': 5}
        self.HDBSCAN_hyperparams = HDBSCAN_hyperparams  # {}

        # 更新 HDBSCAN 的超参数
        self.UMAP_hyperparams["n_components"] = n_dims_umap
        self.UMAP_hyperparams["n_neighbors"] = n_neighbors_umap
        self.UMAP_hyperparams["min_dist"] = min_dist_umap
        self.UMAP_hyperparams["metric"] = metric_umap
        self.UMAP_hyperparams["random_state"] = random_state
        self.UMAP_hyperparams["verbose"] = verbose
        self.umap = umap.UMAP(**self.UMAP_hyperparams)
        #
        self.HDBSCAN_hyperparams["min_cluster_size"] = min_cluster_size_hdbscan
        self.HDBSCAN_hyperparams["metric"] = metric_hdbscan
        self.HDBSCAN_hyperparams["cluster_selection_method"] = cluster_selection_method_hdbscan
        self.number_clusters_hdbscan = number_clusters_hdbscan
        self.hdbscan = hdbscan.HDBSCAN(**self.HDBSCAN_hyperparams)

    def reduce_dimensions_umap(self, embeddings: np.ndarray) -> Tuple[np.ndarray, umap.UMAP]:
        """
        Reduces dimensions of embeddings using UMAP.降维算法

        Args:
            embeddings (np.ndarray): Embeddings to reduce.

        Returns:
            tuple: A tuple containing two items:
                - reduced_embeddings (np.ndarray): Reduced embeddings.
                - umap_mapper (umap.UMAP): UMAP mapper for transforming new embeddings, especially embeddings of the vocabulary. (MAKE SURE TO NORMALIZE EMBEDDINGS AFTER USING THE MAPPER)
        """
        # 使用UMAP的超参数，注意检查超参数
        mapper = umap.UMAP(**self.UMAP_hyperparams).fit(embeddings)  # embeddings: [document_num, hidden_size]
        dim_red_embeddings = mapper.transform(embeddings)  # 开始降维 [document_num,dimension], eg: [23,5]
        dim_red_embeddings = dim_red_embeddings / np.linalg.norm(dim_red_embeddings, axis=1).reshape(-1, 1)
        return dim_red_embeddings, mapper

    def cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.
        使用HDBSCAN对降维后的嵌入进行聚类，如果指定了固定的聚类数量，则进一步使用层次聚类，并重新索引标签使其连续。
        Args:
            embeddings (np.ndarray): Embeddings to cluster.

        Returns:
            np.ndarray: Cluster labels.
        """
        ## 使用 HDBSCAN 进行聚类
        labels = self.hdbscan.fit_predict(embeddings)
        if np.all(labels == -1):
            print(f"聚类后所有数据都是离群点，请检查数据或调整参数")
            raise Exception(f"聚类后所有数据都是离群点，请检查数据或调整参数")
        outliers = np.where(labels == -1)[0]
        ## 找到所有的离群点（标签为 -1）
        if self.number_clusters_hdbscan is not None:
            clusterer = AgglomerativeClustering(
                n_clusters=self.number_clusters_hdbscan)  # # 使用层次聚类重新聚类  #one cluster for outliers
            labels = clusterer.fit_predict(embeddings)
            labels[outliers] = -1  # # 将离群点的标签重新设置为 -1

        # reindex to make the labels consecutive numbers from -1 to the number of clusters. -1 is reserved for outliers
        unique_labels = np.unique(labels)
        unique_labels_no_outliers = unique_labels[unique_labels != -1]
        map2newlabel = {label: i for i, label in enumerate(unique_labels_no_outliers)}
        map2newlabel[-1] = -1
        labels = np.array([map2newlabel[label] for label in labels])

        return labels

    def cluster_and_reduce(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, umap.UMAP]:
        """
        该方法首先使用 UMAP 对输入嵌入进行降维，然后使用 HDBSCAN 对降维后的嵌入进行聚类。
        最终返回降维后的嵌入、聚类标签和 UMAP 映射器。
        Args:
            embeddings (np.ndarray): Embeddings to cluster and reduce.

        Returns:
            tuple: A tuple containing three items:
                - reduced_embeddings (np.ndarray): Reduced embeddings.
                - cluster_labels (np.ndarray): Cluster labels.
                - umap_mapper (umap.UMAP): UMAP mapper for transforming new embeddings, especially embeddings of the vocabulary. (MAKE SURE TO NORMALIZE EMBEDDINGS AFTER USING THE MAPPER)
        """

        dim_red_embeddings, umap_mapper = self.reduce_dimensions_umap(embeddings)
        clusters = self.cluster_hdbscan(dim_red_embeddings)  # clusters是聚类的类别信息
        return dim_red_embeddings, clusters, umap_mapper

    def visualize_clusters_static(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Reduce dimensionality with UMAP to two dimensions and plot the clusters.
        使用UMAP将嵌入降维到二维，并使用Matplotlib绘制聚类结果。
        不同的聚类标签对应不同的颜色，离群点（标签为 -1）显示为灰色。
            使用 UMAP 将高维嵌入降维到二维。
            创建颜色调色板，根据标签映射颜色。
            为每个标签（聚类和离群点）绘制散点图。
            将离群点（标签为 -1）显示为灰色。
            显示图例并展示图像。
        Args:
            embeddings (np.ndarray): Embeddings for which to plot clustering.
            labels (np.ndarray): Cluster labels.
        """
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_components=2, random_state=self.random_state, n_neighbors=30, metric="cosine", min_dist=0)
        embeddings_2d = reducer.fit_transform(embeddings)

        # Create a color palette, then map the labels to the colors.
        # We add one to the number of unique labels to account for the noise points labelled as -1.
        palette = plt.cm.get_cmap("tab20", len(np.unique(labels)) + 1)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))

        outlier_shown_in_legend = False

        # Iterate through all unique labels (clusters and outliers)
        for label in np.unique(labels):
            # Find the embeddings that are part of this cluster
            cluster_points = embeddings_2d[labels == label]

            # If label is -1, these are outliers. We want to display them in grey.
            if label == -1:
                color = 'grey'
                if not outlier_shown_in_legend:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label='outlier', s=0.1)
                    outlier_shown_in_legend = True
                else:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=0.1)
            else:
                color = palette(label)
                # Plot the points in this cluster without a label to prevent them from showing up in the legend
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=0.1)

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()

    def visualize_clusters_dynamic(self, embeddings: np.ndarray, labels: np.ndarray, texts: list[str],
                                   class_names: list[str] = None):
        """
        此方法使用 Plotly 进行动态聚类可视化，并支持悬停显示文本信息。会自动弹出1个网页，显示绘图结果
        UMAP 将嵌入降维到二维，然后在二维平面上绘制聚类结果，点的颜色表示聚类标签。
            使用 UMAP 将高维嵌入降维到二维。
            创建一个包含二维坐标和文本信息的数据框（DataFrame）。
            为不同的聚类标签分配颜色，并处理离群点。
            使用 Plotly 绘制散点图，悬停时显示文本信息。
            设置图像的布局和标题，并展示图像。
        Args:
            embeddings (np.ndarray): [document_num, embedding_dim] 文档的嵌入向量。
            labels (np.ndarray): 每个文档对应的聚类标签。eg: [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
            texts (list[str]):  文档的文本内容。
            class_names (list[str], optional): 聚类标签的名称。eg: ['Topic 0: \n"Convenient meal delivery service"\n', 'Topic 1: \nTitle: Online Food Ordering Convenience\n']
        """

        # Reduce dimensionality with UMAP, 可视化的时后，30个类，是不是有些问题, 而且不能大于文本的数量
        reducer = umap.UMAP(n_components=2, random_state=self.random_state, n_neighbors=self.UMAP_hyperparams["n_neighbors"], metric="cosine", min_dist=0)
        embeddings_2d = reducer.fit_transform(embeddings)  #[document_num, 2]

        df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        df['text'] = [text[:200] for text in texts]
        df["class"] = labels

        if class_names is not None:
            df["class"] = [class_names[label] for label in labels]

        # 创建颜色画板，聚类的标签放到和画板颜色对应
        # Exclude the outlier (-1) label from color palette assignment
        unique_labels = [label for label in np.unique(labels) if label != -1]
        palette = plt.cm.get_cmap("tab20", len(unique_labels))

        # Create color map
        color_discrete_map = {
            label: 'rgb' + str(tuple(int(val * 255) for val in palette(i)[:3])) if label != -1 else 'grey' for i, label
            in enumerate(unique_labels)}
        color_discrete_map[-1] = 'grey'

        # plot data points where the color represents the class
        fig = px.scatter(df, x='x', y='y', hover_data=['text', 'class'], color='class',
                         color_discrete_map=color_discrete_map)

        fig.update_traces(mode='markers', marker=dict(size=3))  # Optional: Increase the marker size

        # make plot quadratic
        fig.update_layout(
            autosize=False,
            width=1500,
            height=1500,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            )
        )
        # set title 
        fig.update_layout(title_text='UMAP projection of the document embeddings', title_x=0.5)

        # show plot
        fig.show()

    def umap_diagnostics(self, embeddings, hammer_edges=False):
        """
        该方法用于生成 UMAP 的诊断图，以帮助检查降维效果。
        包括点连接图、PCA 诊断图和局部维度诊断图等，支持使用锤式边捆绑进行连接图的绘制（较为计算密集）。
        Params:
        ------
        embeddings : array-like
            The high-dimensional data for UMAP to reduce and visualize.
        hammer_edges : bool, default False. Is computationally expensive.
            
        """
        new_hyperparams = deepcopy(self.UMAP_hyperparams)
        new_hyperparams["n_components"] = 2
        mapper = umap.UMAP(**new_hyperparams).fit(embeddings)

        # 1. Connectivity plot with points
        print("UMAP Connectivity Plot with Points")
        umap.plot.connectivity(mapper, show_points=True)
        plt.show()

        if hammer_edges:
            # 2. Connectivity plot with edge bundling
            print("UMAP Connectivity Plot with Hammer Edge Bundling")
            umap.plot.connectivity(mapper, edge_bundling='hammer')
            plt.show()

        # 3. PCA diagnostic plot
        print("UMAP PCA Diagnostic Plot")
        umap.plot.diagnostic(mapper, diagnostic_type='pca')
        plt.show()

        # 4. Local dimension diagnostic plot
        print("UMAP Local Dimension Diagnostic Plot")
        umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
        plt.show()
