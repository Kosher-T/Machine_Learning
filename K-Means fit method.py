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

    def fit(self, X):
        """
        Runs k-means until centroids move < tol or max_iter reached.
        Saves final centroids in self.centroids.
        """
        for iteration in range(self.max_iter):
            clusters = self.find_cluster_centroid(X)

            new_centroids = []
            shifts = []

            # 4) Compute new centroids & track shifts
            for old_cent in self.centroids:
                pts = clusters[tuple(old_cent)]
                # avoid empty cluster
                if not pts:
                    new_centroids.append(old_cent)
                    shifts.append(0)
                    continue
                # mean of each dimension
                dim_means = [sum(dim_vals)/len(pts)
                             for dim_vals in zip(*pts)]
                new_centroids.append(dim_means)
                shifts.append(euclidian_distance(dim_means, old_cent))

            # 5) Check for convergence
            max_shift = max(shifts)
            if max_shift < self.tol:
                break

            # 6) Update centroids & loop
            self.centroids = deepcopy(new_centroids)

        print("Centroids:", self.centroids)

    def predict(self, X):
        """
        For each x in X, returns the index of the nearest centroid.
        """
        labels = []
        for p in X:
            dists = [euclidian_distance(p, c) for c in self.centroids]
            labels.append(dists.index(min(dists)))
        return labels

if __name__ == "__main__":
    kmeans = KMeans(3)
    pts = [
        [8, 43],[67, 48],[43, 80],[16, 87],[5, 42],
        [54, 11],[21, 56],[65, 74],[74, 67],[65, 62]
    ]
    kmeans.fit(pts)
    # Output should be "Centroids: [[62.8, 66.2], [12.5, 57.0], [54.0, 11.0]]""
    print("Labels:", kmeans.predict(pts))