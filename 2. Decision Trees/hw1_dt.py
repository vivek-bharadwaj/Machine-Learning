import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splittable is false when all features belongs to one class
        if len(np.unique(labels)) < 2 or len(np.unique(self.features)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

        self.class_label_count_map = dict()
        self.accurate_classifications_count = 0

    # TODO: try to split current node
    def split(self):
        # find branches encoding
        # Find Info Gain for each feature
        # Take Max IG and split
        # When split: features = features - feature_col, labels = labels - removed feature labels
        if self.splittable == True:
            entropy = Util.branch_entropy(self.labels)
            # print(entropy)
            labels_arr = np.array(self.labels)
            unique_labels = np.unique(labels_arr)   # unique labels
            y_map = Util.get_label_enum_map(unique_labels)
            # print(y_map)
            y_T = np.transpose(labels_arr)
            X_T = np.transpose(np.array(self.features))
            # print(X_T[0])

            branches = []
            x_map = []

            for i in range(len(X_T)):
                unique_x, u_count = np.unique(X_T[i], return_counts=True)
                x_map.append(Util.get_label_enum_map(unique_x))
                # print(x_map)
                print("Unique x", unique_x)

                branch = []
                for j in range(len(unique_x)):
                    branch.append(get_count(unique_x[j], X_T[i], y_T))
                branches.append(branch)
            # print(x_map)
            IG = []
            only_IG = []
            for i in range(len(branches)):
                ig = Util.Information_Gain(entropy, branches[i])
                unique_x_vals = len(np.unique(X_T[i]))
                # print(branches[i])
                only_IG.append(ig)
                IG.append((ig, unique_x_vals, i))
            # print(IG)
            only_IG = np.array(only_IG)
            if all(only_IG == 0.0):
                self.splittable = False
                return

            sorted_IG = sorted(IG, key=lambda a: (-a[0], -a[1], a[2]))

            print("Sorted IG", sorted_IG)
            ig_map = {}
            # Creating dict to keep track of IG to index since we need index after sorting
            for i in range(len(IG)):
                ig_map[i] = IG[i]

            # sorted_ig = sorted(ig_map.items(), key=lambda x: -x[1])
            # print("Sorted IG: ", sorted_ig)
            # feature_index = [x[0] for x in sorted_ig]

            self.dim_split = sorted_IG[0][2]
            print(self.dim_split)
            # child_features = np.delete(self.features, self.dim_split, 1)
            # print(child_features)

            self.feature_uniq_split = []

            # Get the next best feature index to expand
            # Get unique values in the feature
            # Loop through unique values

            # print(X_T[self.dim_split])
            unique_x = np.unique(X_T[self.dim_split])
            # unique_x = unique_x.tolist()
            self.feature_uniq_split.extend(unique_x)
            # print(unique_x)

            # print("dim split", self.features[self.dim_split])
            print(self.feature_uniq_split)

            for key in unique_x:
                new_x, new_y, unique_counts = choose_row_by_key(key, self.dim_split, self.features, self.labels)
                child = TreeNode(features=new_x, labels=new_y, num_cls=unique_counts)
                child.split()
                self.children.append(child)

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable == True:
            x_i = None
            x_vals = self.feature_uniq_split
            # print(x_map)
            for i in range(len(x_vals)):
                if x_vals[i] == feature[self.dim_split]:
                    x_i = i
                    break
            if x_i is not None:
                idx_tbd = self.dim_split
                new_features = np.delete(feature, idx_tbd)
                new_features_list = new_features.tolist()
                return self.children[x_i].predict(new_features_list)

        else:
            return self.cls_max

    def populate_classification_label_attrs(self, X_test, y_test_label):
        if self.splittable == True:
            X_i = None
            dim_idx = self.dim_split
            X_val = X_test[dim_idx]
            uniq_xs = self.feature_uniq_split
            for i in range(len(uniq_xs)):
                if uniq_xs[i] == X_val:
                    X_i = i
                    break

            if X_i is None:
                self.class_label_count_map[y_test_label] = self.class_label_count_map.get(y_test_label, 0) + 1
                if y_test_label == self.cls_max:
                    self.accurate_classifications_count += 1
            else:
                new_x_test = np.delete(X_test, self.dim_split).tolist()
                self.children[X_i].populate_classification_label_attrs(new_x_test, y_test_label)

        else:
            self.class_label_count_map[y_test_label] = self.class_label_count_map.get(y_test_label, 0) + 1
            if y_test_label == self.cls_max:
                self.accurate_classifications_count += 1

        print(self.accurate_classifications_count, self.class_label_count_map)
    #

    def reduced_error_pruning_helper(self):
        if self.splittable == True:
            prev_accuracy = self.accurate_classifications_count

            for child in self.children:
                class_labels, accuracy = child.reduced_error_pruning_helper()
                prev_accuracy += accuracy

                for cls, cls_ct in class_labels.items():
                    self.class_label_count_map[cls] = self.class_label_count_map.get(cls, 0) + cls_ct

            if len(self.class_label_count_map) != 0:
                new_accuracy = 0
                for label in self.class_label_count_map:
                    if label == self.cls_max:
                        new_accuracy += self.class_label_count_map[label]

                if prev_accuracy <= new_accuracy:
                    self.splittable = False
                    self.children = []
                    self.accurate_classifications_count = new_accuracy

                else:
                    self.accurate_classifications_count = prev_accuracy

        return self.class_label_count_map, self.accurate_classifications_count


def get_count(val, feat, label):
    sol = [0]*len(np.unique(label))
    # translate label
    print("Label", label)
    unique_l = np.unique(label)
    y_map = Util.get_label_enum_map(unique_l)
    print("Y Map", y_map)
    y_translate = [0] * len(label)
    for i in range(len(label)):
        y_translate[i] = y_map[label[i]]
    print("Y translate", y_translate)
    for i in zip(feat, y_translate):
        if i[0] == val:
            sol[i[1]] += 1
    #sol = list(filter(lambda x: x != 0, sol))
    print(sol)
    return sol


def choose_row_by_key(key, feature_index, feature, y):
    new_x = []
    new_y = []

    for i in range(len(feature)):
        if feature[i][feature_index] == key:
            new_x.append(feature[i])
            new_y.append(y[i])
    new_x = np.array(new_x)
    new_x = np.delete(new_x, feature_index, 1)
    new_y = np.array(new_y)
    _, unique_count = np.unique(new_y, return_counts=True)

    # print(new_x, new_x.shape)
    # print(new_y)
    return new_x.tolist(), new_y.tolist(), len(unique_count)


# if __name__ == '__main__':
#     dt = DecisionTree()
#     # features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
#     # labels = [0, 0, 1, 1]
#     x = [['b','b','b','b','c','c'], ['a','a','c','c','c','c']]
#     y = [1,2]
#     dt.train(x, y)
