import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    h = []
    for branch in branches:
        s = np.sum(branch)
        p = branch / s
        p = [i * np.log2(i, out=np.zeros_like(i), where=(i != 0)) for i in p]
        # print(p)
        e = -np.sum(p)
        # print(e)
        # print("Sum: ", np.sum(branches))
        # print("e*s", e * s)
        h.append(e * s / get_sum(branches))

    E = np.sum(h)
    return S - E


def get_sum(l):
    s = 0
    for i in l:
        s += sum(i)
    return s


def branch_entropy(labels):
    labels, label_counts = np.unique(labels, return_counts=True)
    label_counts = label_counts / np.sum(label_counts)
    label_counts = [i * np.log2(i) for i in label_counts]
    return -1 * np.sum(label_counts)


def get_label_enum_map(unique_labels):
    unique_dict = {}
    i = 0
    for key in unique_labels:
        unique_dict[key] = i
        i += 1
    return unique_dict


# TODO: implement reduced error pruning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    root_node = decisionTree.root_node
    for example_row_id in range(len(X_test)):
        # print("Index: ", example_row_id)
        # print("XTest[row_id]: ", X_test[example_row_id])
        root_node.populate_classification_label_attrs(X_test[example_row_id], y_test[example_row_id])   # Count error and populate error dictionary in TreeNode class
    root_node.reduced_error_pruning_helper()


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
