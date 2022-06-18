from sklearn_extra.cluster import KMedoids

def run_k_medoids(data, n_clusters):
    model = KMedoids(n_clusters=n_clusters)
    model.fit(data)
    centroids = model.medoid_indices_
    return model.labels_, centroids

def get_centroids_clusters(data,centroids):
    labels = []
    for i in  range(len(data)):
        if data[i] in centroids:
            labels.append(data[i])
    return labels
