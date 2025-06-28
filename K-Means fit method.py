from copy import deepcopy

def euclidian_distance(a, b):
    return sum((a_i - b_i)**2 for a_i, b_i in zip(a, b))**0.5

class KMeans:
    def __init__(self, k, tol=0.01, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = []

    def find_cluster_centroid(self, points):
        """
        Assigns each point in `points` to the nearest centroid.
        Returns a dict mapping each centroid (tuple) -> list of assigned points.
        If self.centroids is empty, picks the 3rd,5th,7th… points as init.
        """
        # 1) Initialize if first call
        if not self.centroids:
            init = []
            for i in range(2, len(points), 2):    # 3rd,5th,7th...
                init.append(points[i])
                if len(init) == self.k:
                    break
            self.centroids = deepcopy(init)

        # 2) Build empty clusters
        clusters = {tuple(c): [] for c in self.centroids}

        # 3) Assign points
        for p in points:
            # find closest centroid
            dists = [euclidian_distance(p, c) for c in self.centroids]
            idx = dists.index(min(dists))
            clusters[tuple(self.centroids[idx])].append(p)

        return clusters


    def fit(self, X_train):
        # Write you code here
        def cluster_centroid(dataset: list) -> list:
            '''
            # This function will find the cluster centroid for the given training data
            '''
            return [sum(coord) / len(new_cluster) for coord in zip(*dataset)]

        cluster = self.find_cluster_centroid(X_train)
        # cluster = {
        #           (43, 80): [[67, 48], [43, 80], [16, 87], [65, 74], [74, 67], [65, 62]],
        #           (5, 42): [[8, 43], [5, 42]],
        #           (21, 56): [[54, 11], [21, 56]]
        #           }

        dista = True

        while dista:
            updates = {}
            max_shift = 0
            for centroid, new_cluster in cluster.items():
                new_centroid = cluster_centroid(new_cluster)
                distance = euclidian_distance(new_centroid, centroid)
                max_shift = max(max_shift, distance)
                updates[tuple(new_centroid)] = new_cluster
            if max_shift < 0.01:
                dista = False
            else:
                cluster.clear()
                cluster.update(updates)

        self.centroids = list(cluster.keys())
        print("Centroids:", self.centroids)
    
    def predict(self, X_test):
        pass


if __name__ == "__main__":
    kmeans = KMeans(3)
    cluster_points = [[8, 43], [67, 48], [43, 80], [16, 87], [5, 42], [54, 11], [21, 56], [65, 74], [74, 67], [65, 62]]
    kmeans.fit(cluster_points)
    # Expected output: latest centroids: [[62.8, 66.2], [12.5, 57.0], [54.0, 11.0]]
