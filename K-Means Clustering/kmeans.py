import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    """
    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    """
    # TODO:
    # implement the K-Means++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    centers = []
    c = generator.randint(n)
    centers.append(c)

    for i in range(n_cluster - 1):
        distances = squared_distance(x, centers)
        distances_sum = np.sum(distances)
        probability = distances / distances_sum
        next_center = np.argmax(probability)
        centers.append(next_center)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers


def squared_distance(x, c):
    """
    :param x: Points
    :param c: Centers
    :return: Distances
    """
    distances = []
    nearest_center = x[c[-1]]

    for i in range(len(x)):
        point = x[i]
        distances.append(np.linalg.norm(point - nearest_center) ** 2)

    return np.asarray(distances)


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    """
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for k-means clustering (Int)
            max_iter - maximum updates for k-means clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    """

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
        self.centers = None

    def fit(self, x, centroid_func=get_lloyd_k_means):
        """
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or K-means++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's
                assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        """
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DO NOT CHANGE CODE ABOVE THIS LINE

        K = self.n_cluster      # Number of clusters
        centroids = np.zeros((K, D))
        y = np.zeros((N,), dtype=int)

        # Initialize step
        centroids = x[np.random.choice(N, self.n_cluster, replace=True)]     # Means set randomly
        J = 10 ** 10
        i = 0

        # Repeat step
        for i in range(self.max_iter):
            # Compute L2 norm squared distance between each point and centroids by transforming centroids to 3-D
            distance_to_centroids = np.array((x - centroids[:, None]) ** 2)     # New shape: (K, N, D)
            sum_of_squares = np.sum(distance_to_centroids, axis=2)          # Compute sum along 3rd axis, shape: (K, N)
            assignments = np.argmin(sum_of_squares, axis=0)                 # y
            y = assignments

            # Compute Distortion J_new
            inner_sum = [np.sum((x[assignments == k] - centroids[k]) ** 2) for k in range(K)]
            outer_sum = np.divide(np.sum(inner_sum), N)
            J_New = outer_sum

            if np.abs(J - J_New) <= self.e:
                break

            # Set J := J_New
            J = J_New

            # Compute Centroids (mean)
            with np.errstate(divide='ignore'):
                means = np.array([np.nanmean(np.transpose(x[y == k]), axis=1) for k in range(K)])
                nan_index = np.where(np.isnan(means))
                means[nan_index] = centroids[nan_index]
                centroids = means

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, i


class KMeansClassifier():
    """
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    """

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
        self.centroid_labels = None
        self.centroids = None

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        """
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DO NOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, num_iterations = k_means.fit(x)

        centroid_labels = list()
        K = self.n_cluster

        members_of_cluster_k = [y[membership == k] for k in range(K)]
        # print(members_of_cluster_k)
        for cluster in members_of_cluster_k:
            if len(cluster) != 0:
                votes = np.argmax(np.bincount(cluster))
                centroid_labels.append(votes.item())
            else:
                centroid_labels.append(0)
        centroid_labels = np.array(centroid_labels)

        # DO NOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        """
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DO NOT CHANGE CODE ABOVE THIS LINE
        distance_to_centroids = np.array((x - self.centroids[:, None]) ** 2)  # New shape: (K, N, D)
        sum_of_squares = np.sum(distance_to_centroids, axis=2)  # Compute sum along 3rd axis, shape: (K, N)
        assignments = np.argmin(sum_of_squares, axis=0)
        labels = self.centroid_labels[assignments]

        # assignments = np.argmin(np.sum((x - self.centroids[:, None]) ** 2, axis=2), axis=0)
        # labels = self.centroid_labels[assignments]

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    """
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    """

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DO NOT CHANGE CODE ABOVE THIS LINE

    M, N, K = image.shape
    image_2d = image.reshape(M * N, K)
    code_vectors = np.array(code_vectors)

    euclidean_distances = np.sum((image_2d - code_vectors[:, None]) ** 2, axis=2)
    assignments = np.argmin(euclidean_distances, axis=0)
    new_im = code_vectors[assignments]
    new_im = new_im.reshape(M, N, K)

    # DO NOT CHANGE CODE BELOW THIS LINE
    return new_im
