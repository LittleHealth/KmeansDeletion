from sklearn.metrics import silhouette_score, mutual_info_score


def loss_ratio(optimal_kmeans_loss, kmeans_loss):
    return optimal_kmeans_loss / kmeans_loss


def silhouette(data, labels_pred):
    return silhouette_score(data, labels_pred)


def mutual_info(labels_real, labels_pred):
    return mutual_info_score(labels_real, labels_pred)
