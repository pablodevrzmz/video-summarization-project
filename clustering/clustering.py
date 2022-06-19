from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from numpy import indices, unique, where, array_equal

def run_k_medoids(data, n_clusters):
    model = KMedoids(n_clusters=n_clusters)
    model.fit(data)
    centroids = model.medoid_indices_
    return model.labels_, centroids

def run_dbscan(data, eps=0.5, min_samples=5):
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_
    return __identify_centroids(data,clusters)

def run_aglomerative_clustering(data, n_clusters):
    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)
    return __identify_centroids(data,clusters)

def get_centroids_clusters(data,centroids):
    labels = []
    for i in  range(len(data)):
        if data[i] in centroids:
            labels.append(data[i])
    return labels

def __identify_centroids(data,clusters):
    unique_clusters = unique(clusters)
    control = {}
    
    for i in unique_clusters: 
        control[i] = []
    
    for i, v in enumerate(clusters):
        control[v].append(data[i])
    
    centroid_indexes = []
    
    for key in control:
        _,indexes = run_k_medoids(control[key],1)
        assert len(indexes) == 1
        center = control[key][indexes[0]]
        for i,d in enumerate(data):
            if array_equal(d,center):
                centroid_indexes.append(i)

    return clusters, centroid_indexes