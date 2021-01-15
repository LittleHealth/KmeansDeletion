from sklearn.metrics import silhouette_score, mutual_info_score
from DCKmeans import DCKmeans, Kmeans
from QKmeans import QKmeans
from utils import online_deletion_stream

# definition of evaluation tools


def loss_ratio(optimal_kmeans_loss, kmeans_loss):
    return optimal_kmeans_loss / kmeans_loss


def silhouette(data, labels_pred):
    return silhouette_score(data, labels_pred)


def mutual_info(labels_real, labels_pred):
    return mutual_info_score(labels_real, labels_pred)


class evaluation:
    def __init__(self, data, labels, k, DEL_NUM):
        self.data = data
        self.label = labels
        self.k = k
        self.DEL_NUM = DEL_NUM
        self.kmeans = Kmeans(k, 20)
        self.qkmeans = QKmeans(k, 0.5)
        self.dckmeans = DCKmeans([k, k], [1, 16])
        self.centers = self.assignments = self.loss = 0
        self.qcenters = self.qassignments = self.qloss = 0
        self.dccenters = self.dcassignments = self.dcloss = 0

    def run(self):
        self.centers, self.assignments, self.loss = self.kmeans.run(self.data)
        self.qcenters, self.qassignments, self.qloss = self.qkmeans.run(self.data)
        self.dccenters, self.dcassignments, self.dcloss = self.dckmeans.run(self.data, True)

    # 测试三种方法的基础性能
    def test_basic(self):
        print("kmeans loss ratio: ", 1)
        loss_ratio_q = loss_ratio(self.qloss, self.loss)
        print("qkmeans loss ratio: ", loss_ratio_q)
        loss_ratio_dc = loss_ratio(self.dcloss, self.loss)
        print("dckmeans loss ratio: ", loss_ratio_dc)
        print()

        silh = silhouette(self.data, self.assignments)
        print("kmeans silhouette score: ", silh)

        qsilhouette = silhouette(self.data, self.qassignments)
        print("qkmeans silhouette score: ", qsilhouette)

        dcsilhouette = silhouette(self.data, self.dcassignments)
        print("dckmeans silhouette score: ", dcsilhouette)
        print()

        mutual = mutual_info(self.label, self.assignments)
        print("kmeans mutual info score: ", mutual)

        qmutual = mutual_info(self.label, self.qassignments)
        print("qkmeans mutual info score: ", qmutual)

        dcmutual = mutual_info(self.label, self.dcassignments)
        print("dckmeans mutual info score: ", dcmutual)

    # 测试删除效果
    def test_deletion(self):
        #  qkmeans删除
        online_deletion_stream(self.DEL_NUM, self.qkmeans, False)
        deleted_data_q = self.qkmeans.data

        qcenters = self.qkmeans.centroids_res
        qassignments = self.qkmeans.clusters_res
        qloss = self.qkmeans.min_loss

        # 删除后的结果
        deleted_loss_q = qloss / self.loss
        deleted_sil_q = silhouette(deleted_data_q, qassignments)

        # 重新运行的结果
        qcenters, qassignments, qloss = self.qkmeans.run(deleted_data_q)
        rerun_loss_q = qloss / self.loss
        rerun_sil_q = silhouette(deleted_data_q, qassignments)

        print("deleted loss of qkmeans: ", deleted_loss_q)
        print("rerun loss of qkmeans: ", rerun_loss_q)
        print()
        print("deleted silhouette of qkmeans: ", deleted_sil_q)
        print("rerun silhouette of qkmeans: ", rerun_sil_q)
        print()

        # dckmeans删除

        online_deletion_stream(self.DEL_NUM, self.dckmeans, False)
        deleted_data_dc = self.dckmeans.data

        dccenters = self.dckmeans.centers
        dcassignments = self.dckmeans.assignments
        dcloss = self.dckmeans.loss
        _, _, loss = self.kmeans.run(deleted_data_dc)

        # 删除后的结果
        deleted_loss_dc = dcloss / loss
        deleted_sil_dc = silhouette(deleted_data_dc, dcassignments)

        # 重新运行的结果
        dccenters, dcassignments, dcloss = self.dckmeans.run(deleted_data_dc, True)
        rerun_loss_dc = dcloss / loss
        rerun_sil_dc = silhouette(deleted_data_dc, dcassignments)

        print("deleted loss of dckmeans: ", deleted_loss_dc)
        print("rerun loss of dckmeans: ", rerun_loss_dc)
        print()
        print("deleted silhouette of dckmeans: ", deleted_sil_dc)
        print("rerun silhouette of dckmeans: ", rerun_sil_dc)

    # 测试删除时间
    def test_deletion_time(self):
        print('Simulation deletion stream for kmeans')
        online_deletion_stream(self.DEL_NUM, self.kmeans, True)
        print()

        print('Simulation deletion stream for qkmeans')
        online_deletion_stream(self.DEL_NUM, self.qkmeans, True)
        print()

        print('Simulation deletion stream for dckmeans')
        online_deletion_stream(self.DEL_NUM, self.dckmeans, True)
