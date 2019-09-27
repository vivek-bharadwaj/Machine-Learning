import numpy as np
from collections import Counter


def get_majority_label(point):
    count = Counter(point)
    return max(count, key=count.get)


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.training_features = None
        self.training_labels = None

    # TODO: save features and label to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.training_features = features
        self.training_labels = labels
        #raise NotImplementedError

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        labels = []
        # majority_labels = []
        for test_point in features:
            k_nearest_neighbors_labels = self.get_k_neighbors(test_point)
            labels.append(get_majority_label(k_nearest_neighbors_labels))
        # labels = [tuple(i) for i in labels]

        # for i in range(len(labels)):
        #     label = get_majority_label(labels[i])
        #     majority_labels.append(label)
        # print(majority_labels)
        # print(labels)
        return labels

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighbors.
        :param point: List[float]
        :return:  List[int]
        """
        distances = []
        for training_point, label in zip(self.training_features, self.training_labels):
            distance = self.distance_function(training_point, point)
            distances.append((distance, label))
        distances.sort(key=lambda x: x[0])

        # print(distances)
        # print(self.k)
        distances = distances[:self.k]
        k_nearest_neighbors_labels = [int(x[1]) for x in distances]
        # print("K nearest neighbors list: ", k_nearest_neighbors_labels)
        return k_nearest_neighbors_labels


if __name__ == '__main__':
    print(np.__version__)
