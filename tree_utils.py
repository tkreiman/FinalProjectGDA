from sys import path_importer_cache
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
def build_newick_from_dt(clf, add_leaf_class_edges = False, match_class_leaves = False):
    global curr_leaf_num
    curr_leaf_num = 0
    children_left = clf.tree_.children_left #children_left[i] gives the id of the left child of node i
    children_right = clf.tree_.children_right
    #print("gini: {}".format(clf.tree_.impurity))
    if match_class_leaves:
        leaf_node_ids = [i for i in range(len(clf.tree_.children_left)) if clf.tree_.children_left[i] == -1]
        def leaf_purity_sort_func(e):
            return clf.tree_.impurity[e]
        leaf_node_ids.sort(key=leaf_purity_sort_func)
        leaf_class_mapping = {}
        #print("leaf node ids: {}".format(leaf_node_ids))
        for leaf_id in leaf_node_ids:
            for _ in range(len(clf.tree_.value[leaf_id][0])):
                leaf_class = np.argmax(clf.tree_.value[leaf_id][0])
                if leaf_class not in leaf_class_mapping.values():
                    leaf_class_mapping[leaf_id] = leaf_class
                    break
                else:
                    #print("leaf class max issue: {}".format(clf.tree_.value[leaf_id][0]))
                    clf.tree_.value[leaf_id][0][leaf_class] = 0
                    continue
        bad_leaf_count = 0
        for leaf_id in leaf_node_ids:
            if leaf_id not in leaf_class_mapping.keys():
                bad_leaf_count += 1
                for i in range(len(clf.classes_)):
                    if i not in leaf_class_mapping.values():
                        leaf_class_mapping[leaf_id] = i
        print("this decision tree has {} bad leaves".format(bad_leaf_count))

    #print("class labels: {}".format(clf.classes_))
    
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
            elif match_class_leaves:
                return "{}:{}".format(leaf_class_mapping[node_id], lengths[node_id])
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
        nw = build_newick_from_dt(dt, match_class_leaves=True)
        forest.append([nw, dt, dt.score(Xtst,ytst)])
    return forest

# def newick_to_sklearn(newick_str, num_leaves):
#     clf = tree.DecisionTreeClassifier(max_leaf_nodes=num_leaves)




def write_to_tree_dist_program_input():
    forest_test = find_good_trees(50)
    nws, trees, scores = zip(*forest_test)
    print(min(scores), max(scores))
    scores = np.array(scores)
    sorted_idxs = np.argsort(scores)
    bad_idxs = sorted_idxs[:5]
    good_idxs = sorted_idxs[-5:]
    print("bad scores:{}".format(scores[bad_idxs]))
    print("good scores:{}".format(scores[good_idxs]))
    with open("../gtp_170317/example/dt_no_class_edges_3", "w") as f:
        for i in range(len(good_idxs)):
            f.write(nws[good_idxs[i]])
            f.write('\n')
        for i in range(len(bad_idxs)):
            f.write(nws[bad_idxs[i]])
            f.write('\n')


    
def analyze_tree_dists(filename):
    dist_matrix = np.zeros((10, 10))
    with open(filename, "r") as f:
        for line in f:
            if line == "\n":
                continue
            line = line.split("\t")
            t1 = int(line[0])
            t2 = int(line[1])
            dist = float(line[2])
            dist_matrix[t1][t2] = dist
            dist_matrix[t2][t1] = dist
    print(dist_matrix)
    intra_good_tree_dist = 0
    good_to_bad_tree_dist = 0
    for i in range(5):
        for j in range(5):
            intra_good_tree_dist += dist_matrix[i][j]
        for j in range(5, 10):
            good_to_bad_tree_dist += dist_matrix[i][j]
    intra_good_tree_dist /= 20
    good_to_bad_tree_dist /= 25
    return intra_good_tree_dist, good_to_bad_tree_dist


def random_tree_fromdata(X, y):
    # random gaussian array of shape X.shape with values between pix_min and pix_max
    X_rand = np.random.randn(*X.shape) * 3

    # random list of digits between 0 and 9
    y_rand = np.random.randint(0, 10, y.shape[0])
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)
    clf = clf.fit(X_rand,y_rand)

    nw = build_newick_from_dt(clf, match_class_leaves=True)
    return nw

X, y, DATA_MIN, DATA_MAX = load_data()
NUM_TRAIN = int(len(X) * 0.50)
print(random_tree_fromdata(X, y))
# write_to_tree_dist_program_input()

#path_to_output_file = "../gtp_170317/outputs/output6.txt"
#good_dist, bad_dist = analyze_tree_dists(path_to_output_file)
#print("avg intra_good_tree_dist:{}, avg good_to_bad_tree_dist: {}".format(good_dist, bad_dist))



