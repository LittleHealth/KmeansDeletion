from DCKmeans import DCKmeans, Kmeans
from QKmeans import QKmeans
import matplotlib.pyplot as plt
import time
import numpy as np


def show_cluster(centers, assignments, data, title=""):
    plt.subplot()
    plt.scatter(x=data[:, 0], y=data[:, 1], c=assignments)
    plt.scatter(x=centers[:, 0], y=centers[:, 1], c='k', marker='+', s=180)
    plt.title(title)
    plt.show()


def sample_gmm(params):
    clusters = []
    for param in params:
        cluster = np.random.multivariate_normal(mean=param["mean"], cov=param["cov"], size=param["size"])
        clusters.extend(cluster)
    return np.array(clusters)


def online_deletion_stream(num_dels, model):
    t0 = time.time()
    c = 1
    for _ in range(num_dels):
        dr = np.random.choice(model.n, size=1)[0]
        if _ % 10 == 0:
            print(f'processing deletion request # {c}...')
        model.delete(dr)
        c += 1
    t = time.time()
    print(f'Total time to process {c - 1} deletions is {t - t0}')


N = 10000
# covariance = [[1, 0.1], [0.1, 1]]
covariance = [[0.05, 0], [0, 0.05]]
# clusters_params = [{"mean": [-5, 5], "cov": covariance, "size": 5 * N},
#                    {"mean": [0, 0], "cov": covariance, "size": 5 * N},
#                    {"mean": [5, 5], "cov": covariance, "size": 5 * N},
#                    {"mean": [5, -5], "cov": covariance, "size": 5 * N},
#                    {"mean": [-5, -5], "cov": covariance, "size": 8 * N}]
clusters_params = [{"mean": [2, 0], "cov": covariance, "size": 5 * N},
                   {"mean": [0, 2], "cov": covariance, "size": 5 * N},
                   {"mean": [-1.5, -1.5], "cov": covariance, "size": 8 * N}]
data = sample_gmm(clusters_params)
K = 3
# plt.scatter(x=data[:, 0], y=data[:, 1])
kmeans = Kmeans(K, 20)
centers, assignments, _ = kmeans.run(data)
show_cluster(centers, assignments, data, "QKmeans")


dckmeans = DCKmeans([K, K, K], [1, 16, 128])
centers, assignments, _ = dckmeans.run(data, assignment=True)
show_cluster(centers, assignments, data, "DCKmeans")


qkmeans = QKmeans(K, 0.05)
centers, assignments, _ = qkmeans.train(data)
show_cluster(centers, assignments, data, "QKmeans")

DEL_NUM = 100

print('Simulation deletion stream for kmeans')
online_deletion_stream(DEL_NUM, kmeans)

print('Simulation deletion stream for dckmeans')
online_deletion_stream(DEL_NUM, dckmeans)

print('Simulation deletion stream for qkmeans')
online_deletion_stream(DEL_NUM, qkmeans)
