import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import random
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

def load_data():
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X - np.mean(X, axis=0)
    DATA_MIN = np.floor(np.min(X))
    DATA_MAX = np.ceil(np.max(X))
    return X, y, DATA_MIN, DATA_MAX

X, y, DATA_MIN, DATA_MAX = load_data()
NUM_TRAIN = int(len(X) * 0.50)

def bootstrap_sample(n_sets,num_train_samples,len_data):
    data= []
    for i in range(n_sets):
        set_i = random.permutation(len_data)
        train_indices, test_indices = set_i[:num_train_samples], set_i[num_train_samples:] 
        data.append([train_indices, test_indices])
    return data

def get_leaf_classes(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    class_distribution = clf.tree_.value
    leaf_classes = []
    for i in range(n_nodes):
        if children_left[i] == -1:
            leaf_classes.append(np.argmax(class_distribution[i]))
    return leaf_classes

def calc_edge_lengths_from_dt(clf):
    n_nodes = clf.tree_.node_count
    thresholds = clf.tree_.threshold
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    lengths = np.ones(n_nodes)
    for i in range(n_nodes):
        if thresholds[i] != -2:
            left_edge_length = thresholds[i] - DATA_MIN
            right_edge_length = DATA_MAX - thresholds[i] #can use a more sophisticated calc if we want
            lengths[children_left[i]] = left_edge_length
            lengths[children_right[i]] = right_edge_length
    return lengths    

curr_leaf_num = 0
def build_newick_from_dt(clf, add_leaf_class_edges = False):
    global curr_leaf_num
    curr_leaf_num = 0
    children_left = clf.tree_.children_left #children_left[i] gives the id of the left child of node i
    children_right = clf.tree_.children_right
    lengths = calc_edge_lengths_from_dt(clf) #lengths[i] gives the length of the edge to node i from parent
    lengths[0] = 0
    def build_newick_tree_rec(node_id):
        global curr_leaf_num
        left_child = children_left[node_id]
        right_child = children_right[node_id]
        if left_child == -1 and right_child == -1:
            if add_leaf_class_edges:
                class_distribution = clf.tree_.value[node_id][0]
                edge_weights = np.sum(class_distribution) - class_distribution
                leaf_str = "("
                for c in range(len(edge_weights)):
                    unique_class_id = 10 * curr_leaf_num + c 
                    leaf_str += "{}:{}".format(unique_class_id, edge_weights[c])
                    if c != len(edge_weights) - 1:
                        leaf_str += ","
                leaf_str += "):{}".format(lengths[node_id])
                curr_leaf_num  += 1
                return leaf_str     
            else:
                return "{}:{}".format(node_id, lengths[node_id])
        else:
            left_str = build_newick_tree_rec(left_child)
            right_str = build_newick_tree_rec(right_child)
            if node_id == 0:
                return "({},{});".format(left_str, right_str)
            else:
                return "({},{}):{}".format(left_str, right_str, lengths[node_id])
                
    return build_newick_tree_rec(node_id = 0)

def find_good_trees(bootstrap_number):
    indices = bootstrap_sample(bootstrap_number,NUM_TRAIN,len(X))
    forest = []
    for i in range(bootstrap_number):
        Xtr = X[indices[i][0]]
        ytr = y[indices[i][0]]
        Xtst = X[indices[i][1]]
        ytst = y[indices[i][1]]
        dt = tree.DecisionTreeClassifier(max_leaf_nodes=10)
        dt = dt.fit(Xtr,ytr)
        nw = build_newick_from_dt(dt, add_leaf_class_edges=True)
        forest.append([nw, dt, dt.score(Xtst,ytst)])
    return forest


def write_to_tree_dist_program_input():
    forest_test = find_good_trees(50)
    nws, trees, scores = zip(*forest_test)
    print(min(scores), max(scores))
    scores = np.array(scores)
    sorted_idxs = np.argsort(scores)
    bad_idxs = sorted_idxs[:5]
    good_idxs = sorted_idxs[-5:]
    with open("../gtp_170317/example/dt_performance_test", "w") as f:
        for i in range(len(good_idxs)):
            f.write(nws[good_idxs[i]])
            f.write('\n')
        for i in range(len(bad_idxs)):
            f.write(nws[bad_idxs[i]])
            f.write('\n')
    
