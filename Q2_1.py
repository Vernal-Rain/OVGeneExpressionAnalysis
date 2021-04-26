from numpy import load, array
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def kmeans(data, first_n_genes, clusters):
    x = []
    for i in range(first_n_genes):
        x.append(data[i])
    x = array(x)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(x)
    #len(labels_) = first_n_genes, labels_[n] = cluster for nth gene
    return kmeans.labels_


def plot_kmeans(data, first_n_genes, clusters):
    groups = kmeans(data, first_n_genes, clusters)
    plt.hist(groups, bins=clusters)
    plt.xlabel('cluster')
    plt.ylabel('# genes in cluster')
    plt.title('K-means cluster sizes for k=' + str(clusters))
    plt.show()
    return


def q2_1(file, k):
    with load(file) as data:
        gene_names = data['Gene_Name']
        seq = data['SeqData']
    plot_kmeans(seq, 1000, k)
    return


if __name__ == '__main__':
    file = 'Data1.npz'

    q2_1(file, 10)
    q2_1(file, 20)
    q2_1(file, 50)
