from DCKmeans import DCKmeans, Kmeans
from QKmeans import QKmeans
from utils import *


def main(N=100, K=3):
    # covariance = [[1, 0.1], [0.1, 1]]
    covariance = [[0.5, 0], [0, 0.5]]
    # clusters_params = [{"mean": [-5, 5], "cov": covariance, "size": 5 * N},
    #                    {"mean": [0, 0], "cov": covariance, "size": 5 * N},
    #                    {"mean": [5, 5], "cov": covariance, "size": 5 * N},
    #                    {"mean": [5, -5], "cov": covariance, "size": 5 * N},
    #                    {"mean": [-5, -5], "cov": covariance, "size": 8 * N}]
    clusters_params = [{"mean": [2, 0], "cov": covariance, "size": 5 * N},
                       {"mean": [0, 2], "cov": covariance, "size": 5 * N},
                       {"mean": [-1.5, -1.5], "cov": covariance, "size": 8 * N}]
    data = sample_gmm(clusters_params)
    # plt.scatter(x=data[:, 0], y=data[:, 1])
    kmeans = Kmeans(K, 20)
    centers, assignments, _ = kmeans.run(data)
    show_cluster(centers, assignments, data, "Kmeans")

    dckmeans = DCKmeans([K, K, K], [1, 16, 128])
    centers, assignments, _ = dckmeans.run(data, assignment=True)
    show_cluster(centers, assignments, data, "DCKmeans")

    qkmeans = QKmeans(K, 0.05)
    centers, assignments, _ = qkmeans.run(data)
    show_cluster(centers, assignments, data, "QKmeans")

    DEL_NUM = 10

    print('Simulation deletion stream for kmeans')
    online_deletion_stream(DEL_NUM, kmeans)

    print('Simulation deletion stream for dckmeans')
    online_deletion_stream(DEL_NUM, dckmeans)

    print('Simulation deletion stream for qkmeans')
    online_deletion_stream(DEL_NUM, qkmeans)


if __name__ == "__main__":
    main(1000, 3)
