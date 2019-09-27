import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    mul = sum([(x * y) for x, y in zip(real_labels, predicted_labels)])
    real_sum = sum(real_labels)
    pred_sum = sum(predicted_labels)
    mean_f1 = 2 * mul / (real_sum + pred_sum)
    return mean_f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        minkowski_dist = pow(sum(pow(abs(x - y), 3) for x, y in zip(point1, point2)), 1 / 3)
        # print(minkowski_dist)
        return minkowski_dist

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # dist = np.linalg.norm(point1 - point2)
        # return dist

        euclidean_dist = np.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))
        return euclidean_dist

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        inner_product_distance = sum((x * y) for x, y in zip(point1, point2))
        return inner_product_distance

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        abs_x = sum(x * x for x in point1) ** 0.5
        abs_y = sum(y * y for y in point2) ** 0.5
        cosine_similarity = sum([(x * y) for x, y in zip(point1, point2)]) / (abs_x * abs_y)
        return 1 - cosine_similarity

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        # raise NotImplementedError
        gaussian_kernel_distance = -np.exp(-0.5 * sum((x - y) ** 2 for x, y in zip(point1, point2)))
        return gaussian_kernel_distance


distance_priority_map = {
        "euclidean": 4,
        "minkowski": 3,
        "gaussian": 2,
        "inner_prod": 1,
        "cosine_dist": 0,
    }

scale_priority_map = {
        "normalize": 0,
        "min_max_scale": 1
    }


def breakTieOnDistance(f1_score_with_dist):
    # print(f1_score_with_dist)
    priority_to_dist = {
        0: "cosine_dist",
        1: "inner_prod",
        2: "gaussian",
        3: "minkowski",
        4: "euclidean"
    }

    mapped_f1_scores = []
    for i in range(len(f1_score_with_dist)):
        mapped_f1_scores.append([f1_score_with_dist[i][0], distance_priority_map[f1_score_with_dist[i][1]]])

    # sort based 1st priority is value, second priority is row 2
    result = sorted(mapped_f1_scores, key=lambda x: (-x[0], -x[1]))
    return [result[0][0], priority_to_dist[result[0][1]]]


def break_ties(f1_scores_table):
    priority_to_scale = {
        0: "normalize",
        1: "min_max_scale"
    }

    priority_to_distance = {
        0: "cosine_dist",
        1: "inner_prod",
        2: "gaussian",
        3: "minkowski",
        4: "euclidean"
    }

    mapped_f1_scores = []
    for i in range(len(f1_scores_table)):
        mapped_f1_scores.append([f1_scores_table[i][0], scale_priority_map[f1_scores_table[i][1]],
                                 distance_priority_map[f1_scores_table[i][2]], f1_scores_table[i][3],
                                 f1_scores_table[i][4]])
    # print(mapped_f1_scores)
    result = sorted(mapped_f1_scores, key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    # print([result[0][0], priority_to_scale[0][1], priority_to_distance[0][2], result[0][3], result[0][4]])

    return [result[0][0], priority_to_scale[result[0][1]], priority_to_distance[result[0][2]], result[0][3], result[0][4]]

    # print(result)

class HyperparameterTuner:

    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        best_k = 1
        best_distance_func = "cosine_dist"
        best_f1 = 0
        for k in range(1, min(len(y_train), 30), 2):

            f1_score_with_dist = []  # contains [f1, func_name] for k in this iteration
            model_map = {}
            for name, fn in distance_funcs.items():
                knn_model = KNN(k, fn)
                knn_model.train(x_train, y_train)
                f1_score_val = f1_score(y_val, knn_model.predict(x_val))
                f1_score_with_dist.append([f1_score_val, name])
                model_map[str(k)+"_"+name] = knn_model

            # for the given k val, breakties between the distance function
            curr_f1_val, curr_distance_func = breakTieOnDistance(f1_score_with_dist)

            if curr_f1_val > best_f1:
                best_f1 = curr_f1_val
                best_distance_func = curr_distance_func
                best_k = k
                self.best_model = model_map[str(best_k)+"_"+best_distance_func]
            elif curr_f1_val == best_f1:
                #check for distance function
                if distance_priority_map[curr_distance_func] > distance_priority_map[best_distance_func]:
                    best_f1 = curr_f1_val
                    best_distance_func = curr_distance_func
                    best_k = k
                    self.best_model = model_map[str(best_k)+"_"+best_distance_func]

                # elif (curr_distance_func == best_distance_func):
                #     self.best_model = model_map[str(best_k)+"_"+best_distance_func]
                    # do nothing since the lower k is already assigned

        self.best_k = best_k
        self.best_distance_function = best_distance_func

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and distance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        f1_scores_table = []
        knn_model_map = {}

        for k in range(1, min(len(y_train), 30), 2):
            for d_name, d_fn in distance_funcs.items():
                for sc_name, sc_class in scaling_classes.items():
                    scale = sc_class()
                    x_train_scaled = scale(x_train)
                    x_val_scaled = scale(x_val)
                    knn_model = KNN(k, d_fn)
                    knn_model.train(x_train_scaled, y_train)
                    f1_score_val = f1_score(y_val, knn_model.predict(x_val_scaled))
                    f1_scores_table.append([f1_score_val, sc_name, d_name, k, knn_model])
                    key = f'{k}_{d_name}_{sc_name}'
                    # print(f'Key {key}')
                    knn_model_map[key] = knn_model
        # print(f1_scores_table)

        # print(break_ties(f1_scores_table))
        best_f1, best_scaler, best_distance_func, best_k, best_model = break_ties(f1_scores_table)
        self.best_k = best_k
        self.best_scaler = best_scaler
        self.best_distance_function = best_distance_func
        self.best_model = best_model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            normalized_features = np.true_divide(features, norm)
            normalized_features[normalized_features == np.inf] = 0
            normalized_features = np.nan_to_num(normalized_features)
            return normalized_features


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min = []
        self.max = []

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        X = np.array(features)
        if len(self.min) == 0:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_features = (X - self.min) / (self.max - self.min)
            X_normalized = normalized_features.tolist()
            return X_normalized

