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
        if _ % int(num_dels/10) == 0:
            print(f'processing deletion request # {c}...')
        model.delete(dr)
        c += 1
    t = time.time()
    print(f'Total time to process {c - 1} deletions is {t - t0}')