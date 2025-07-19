import argparse
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset

def KMeans_clustering(embeddings, n_clusters=10):
    """
    使用KMeans对嵌入向量进行聚类
    :param embeddings: 嵌入向量数组
    :param n_clusters: 聚类数目
    :return: 聚类标签
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids


def DBSCAN_clustering(embeddings, eps=0.5, min_samples=5):
    """
    使用DBSCAN对嵌入向量进行聚类
    :param embeddings: 嵌入向量数组
    :param eps: DBSCAN的eps参数
    :param min_samples: DBSCAN的min_samples参数
    :return: 聚类标签
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = dbscan.fit_predict(embeddings)
    return cluster_ids


def Agglomerative_clustering(embeddings, n_clusters=10):
    """
    使用层次聚类对嵌入向量进行聚类
    :param embeddings: 嵌入向量数组
    :param n_clusters: 聚类数目
    :return: 聚类标签
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_ids = clustering.fit_predict(embeddings)
    return cluster_ids


def cluster_embeddings(embeddings, method='kmeans', **kwargs):
    """
    根据指定方法对嵌入向量进行聚类
    :param embeddings: 嵌入向量数组
    :param method: 聚类方法 ('kmeans', 'dbscan', 'agglomerative')
    :param kwargs: 其他参数
    :return: 聚类标签
    """
    if method == 'kmeans':
        return KMeans_clustering(embeddings, **kwargs)
    elif method == 'dbscan':
        return DBSCAN_clustering(embeddings, **kwargs)
    elif method == 'agglomerative':
        return Agglomerative_clustering(embeddings, **kwargs)
    else:
        raise ValueError("Unsupported clustering method: {}".format(method))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/mscoco_emb.parquet')
    parser.add_argument('--cluster_method', type=str, default='agglomerative', choices=['kmeans', 'dbscan', 'agglomerative'])
    parser.add_argument('--output_path', type=str, default='./data/emb_cluster_vis_agglomerative.jsonl')
    parser.add_argument('--n_clusters', type=int, default=15)
    parser.add_argument('--sample_per_cluster', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 1. 读取文件
    emb_path = args.data_path
    dataset = load_dataset("parquet", data_files=emb_path, split="train")
    embeddings = dataset["embedding"]
    embeddings = np.array(embeddings)
    labels = dataset["url"]
    sample_labels_final = []

    # 2. 聚类
    cluster_embeddings_method = args.cluster_method  # 可选 'kmeans', 'dbscan', 'agglomerative'
    n_clusters = args.n_clusters
    if cluster_embeddings_method == 'kmeans':
        cluster_ids = cluster_embeddings(embeddings, method='kmeans', n_clusters=n_clusters)
    elif cluster_embeddings_method == 'dbscan':
        eps = 1
        min_samples = args.n_clusters
        cluster_ids = cluster_embeddings(embeddings, method='dbscan', eps=eps, min_samples=min_samples)
        print(cluster_ids)
        print(len(cluster_ids))
        print(f"DBSCAN found {len(set(cluster_ids))} clusters (including noise).")
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)  # -1 is noise in DBSCAN
    elif cluster_embeddings_method == 'agglomerative':
        cluster_ids = cluster_embeddings(embeddings, method='agglomerative', n_clusters=n_clusters)

    # 3. t-SNE降维
    tsne = TSNE(n_components=2, random_state=args.seed)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. 可视化
    plt.figure(figsize=(10, 8))
    if cluster_embeddings_method == 'dbscan':
        # DBSCAN可能会有噪声点，-1表示噪声
        idx = cluster_ids == -1
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f"Noise, {sum(idx)} samples", alpha=0.6,
                    color='gray')

    for i in range(n_clusters):
        idx = cluster_ids == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f"Cluster {i}, {sum(idx)} samples", alpha=0.6)

        # 在每个类别中，随机采样10个样本
        sample_indices = np.random.choice(np.where(idx)[0], size=min(args.sample_per_cluster, sum(idx)), replace=False)
        # 对应回labels
        sample_labels = [labels[int(j)] for j in sample_indices]
        sample_labels_final.extend(sample_labels)
        # print(f"Cluster {i} sample labels: {sample_labels}")

    plt.legend()
    plt.title("t-SNE Visualization of Text-Image Embeddings Clusters")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    # plt.tight_layout()
    plt.savefig(f"emb_cluster_vis_{cluster_embeddings_method}.png", dpi=200)

    print(len(sample_labels_final))
    filtered_dataset = dataset.filter(lambda item: item["url"] in sample_labels_final)
    print(filtered_dataset)
    filtered_dataset = filtered_dataset.remove_columns(["image", "embedding"])
    filtered_dataset.to_json(args.output_path)