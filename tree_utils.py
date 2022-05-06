import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import random
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

class TreeNode:
    def __init__(self, leafID, children, lengths, binary=False):
        self.leafID = leafID
        self.children = children
        self.lengths = lengths
        self.binary = binary
        self.threshold = None
        self.feature = None
        if self.children is not None or self.lengths is not None:
            assert(len(self.children) == len(self.lengths))
    def get_num_nodes(self):
        num_nodes = 1
        if self.children:
            for i in range(len(self.children)):
                num_nodes += self.children[i].get_num_nodes()
        return num_nodes
    
    def get_leaves(self, leaf_nums=[]):
        if self.children is None:
            leaf_nums.append(self.leafID)
        else:
            for i in range(len(self.children)):
                self.children[i].get_leaves(leaf_nums)
        return leaf_nums

    def get_left_leaves(self):
        if not self.binary:
            raise Exception("tree not binary!")
        if self.children:
            return self.children[0].get_leaves()
        else:
            raise Exception("leaf node has no children")

    def get_right_leaves(self):
        if not self.binary:
            raise Exception("tree not binary!")
        if self.children:
            return self.children[1].get_leaves()
        else:
            raise Exception("leaf node has no children")

    def predict(self, test_pt):
        assert(self.binary)
        if self.children is None:
            return self.leafID #leaf node
        else:
            assert(self.threshold is not None and self.feature is not None)
            if test_pt[self.feature] < self.threshold:
                return self.children[0].predict(test_pt)
            else:
                return self.children[1].predict(test_pt)
    
    def batch_predict(self, X_test):
        classes = np.zeros(len(X_test))
        for i in range(len(X_test)):
            classes[i] = self.predict(X_test[i])
        return classes

    def eval_tree(self, X_test, y_test):
        preds = self.batch_predict(X_test)
        num_right = 0
        for i in range(len(preds)):
            if preds[i] == y_test[i]:
                num_right += 1
        return num_right / len(preds)

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

def find_good_trees(bootstrap_number, random=False):
    forest = []
    if random:
        for i in range(bootstrap_number):
            dt = random_tree_fromdata(X, y)
            nw = build_newick_from_dt(dt, match_class_leaves=True)
            forest.append([nw, dt, dt.score(X,y)])
    else:
        indices = bootstrap_sample(bootstrap_number,NUM_TRAIN,len(X))
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

def get_data_class_rep(X, y, classes):
    class_rep_info = np.zeros((len(classes), X.shape[1]), dtype=X.dtype)
    for i, c in enumerate(classes):
        idxs = np.where(y == c, True, False)
        X_cls = X[idxs]
        class_rep_info[i] = np.mean(X_cls, axis=0)
    return class_rep_info

#def get_mean_tree()

def find_matching_parens_idx(string):
    #print(string)
    stack_height = 1
    for i in range(len(string)):
        #print(stack_height)
        if string[i] == '(':
            stack_height += 1
        elif string[i] == ')':
            stack_height -= 1
        if stack_height == 0:
            return i
        else:
            i += 1
    raise IndexError("no matching parens found!")

def get_len_and_next_idx(newick_str, start_idx):
    j = start_idx
    while j <= len(newick_str):
        if j == len(newick_str) or newick_str[j] == ',':
            length = float(newick_str[start_idx:j])
            return length, j + 1
        else:
            j += 1

def newick_to_tree_rec(newick_str):
    children = []
    lengths = []
    i = 0
    #print(newick_str)
    while i < len(newick_str):
        #print("curr i: {}, curr char: {}".format(i, newick_str[i]))
        if newick_str[i] == "(":
            end_idx = find_matching_parens_idx(newick_str[i+1:])
            #print("i:{}, end idx:{}".format(newick_str[i + 1], end_idx))
            childNode = newick_to_tree_rec(newick_str[i+1:i+1+end_idx])
            children.append(childNode)
            length, next_idx = get_len_and_next_idx(newick_str, i+1+end_idx + 3)
            lengths.append(length)
            i = next_idx
        else:
            leafID = int(newick_str[i])
            leafNode = TreeNode(leafID=leafID,children=None,lengths=None)
            children.append(leafNode)
            #print("curr str: {}".format(newick_str))
            length, next_idx = get_len_and_next_idx(newick_str, i + 2)
            lengths.append(length)
            i = next_idx
    return TreeNode(leafID=-1, children=children, lengths=lengths)
def newick_to_tree(newick_str):
    return newick_to_tree_rec(newick_str[1:-1])

#def find_partition(children, lengths):

def check_tree(node:TreeNode, level=0):
    if node.children is None:
        num_children = 0
    else:
        num_children = len(node.children)
        for i in range(num_children):
            check_tree(node.children[i], level + 1)
    print("level: {}, num_children: {}".format(level, num_children))
    

def convert_tree_to_binary_rec(node:TreeNode):
    if node.children is None:
        node.binary = True
        return node
    elif len(node.children) == 2:
        new_left_node = convert_tree_to_binary_rec(node.children[0])
        new_right_node = convert_tree_to_binary_rec(node.children[1])
        return TreeNode(leafID=-1, children=[new_left_node, new_right_node], lengths=node.lengths, binary=True)
    elif len(node.children) > 2:
        #print("too many children")
        new_left_node = convert_tree_to_binary_rec(node.children[0])
        new_right_node = TreeNode(leafID=-1, children=node.children[1:], lengths=node.lengths[1:])
        new_right_node = convert_tree_to_binary_rec(new_right_node)
        new_children = [new_left_node, new_right_node]
        #total_lengths = node.lengths[0] = sum(new_binary_node.lengths)
        #length1 = node.lengths[0]/ 
        #TODO: might come up with better length reweighing scheme
        new_lengths = [node.lengths[0], np.mean(node.lengths[1:])] 
        #print(len(new_children))
        return TreeNode(leafID=-1, children=new_children, lengths=new_lengths, binary=True)


def process_feat_info(node, class_reps):
    assert(node.binary)
    if node.children is None:
        return
    else:
        thresh = DATA_MIN + (DATA_MAX - DATA_MIN) * (1 - (node.lengths[0]/node.lengths[1]))
        node.threshold = thresh
        left_leaves = node.get_left_leaves()
        right_leaves = node.get_right_leaves()
        print("number of left leaves: {}, right leaves: {}".format(len(left_leaves), len(right_leaves)))
        #get feature that varies the most between left classes and right classes
        left_reps = np.mean(class_reps[left_leaves], axis=0)
        right_reps = np.mean(class_reps[right_leaves], axis=0)
        feat_diff = np.abs(left_reps - right_reps)
        max_diff_feat = np.argmax(feat_diff)
        node.feature = max_diff_feat
        process_feat_info(node.children[0], class_reps)
        process_feat_info(node.children[1], class_reps)
        return
    
def add_pred_info_to_tree(node, X, y):
    classes = np.arange(10)
    class_reps = get_data_class_rep(X=X, y=y, classes=classes)
    process_feat_info(node, class_reps)
    return node
    


# def binary_tree_to_sklearn(node:TreeNode, X, y):
#     num_nodes = node.get_num_nodes()
#     classes = np.arange(10)
#     class_reps = get_data_class_rep(X=X, y=y, classes=classes)
#     clf = tree.DecisionTreeClassifier(max_leaf_nodes=len(class_reps))
#     clf.fit(X[:NUM_TRAIN], y[:NUM_TRAIN])
#     t = clf.tree_
    
#     print("mean_tree_node_count: {}".format(num_nodes))
#     print(t.node_count)
#     t.node_count = num_nodes
#     print(t.node_count)
#     # t.children_right = np.zeros(num_nodes)
#     # t.children_left = np.zeros(num_nodes)
#     # t.feature = np.zeros(num_nodes)
#     # t.threshold = np.zeros(num_nodes)
#     # t.n_node_samples = np.zeros(num_nodes)
#     # t.impurity = np.zeros(num_nodes)


def write_to_tree_dist_program_input(write_all=False, random=False):
    forest_test = find_good_trees(500, random=random)
    nws, trees, scores = zip(*forest_test)
    print(min(scores), max(scores))
    scores = np.array(scores)
    sorted_idxs = np.argsort(scores)
    bad_idxs = sorted_idxs[:5]
    good_idxs = sorted_idxs[-5:]
    print("bad scores:{}".format(scores[bad_idxs]))
    print("good scores:{}".format(scores[good_idxs]))
    with open("../gtp_170317/example/random_trees2", "w") as f:
        if write_all:
            for i in range(len(nws)):
                f.write(nws[i])
                f.write('\n')
        else:
            for i in range(len(good_idxs)):
                f.write(nws[good_idxs[i]])
                f.write('\n')
            for i in range(len(bad_idxs)):
                f.write(nws[bad_idxs[i]])
                f.write('\n')
  
def analyze_tree_dists(filename):
    dist_matrix = np.zeros((10, 10))
    with open(filename, "r") as f:
        num_lines = len(f.readlines())
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
    print("distances to good mean tree from random trees: {}".format(dist_matrix[0]))
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
    #nw = build_newick_from_dt(clf, match_class_leaves=True)
    return clf

X, y, DATA_MIN, DATA_MAX = load_data()
NUM_TRAIN = int(len(X) * 0.50)
# print(random_tree_fromdata(X, y))
classes = np.arange(10)
class_reps = get_data_class_rep(X=X, y=y, classes=classes)
#print("class rep info: {}".format(class_reps))
#write_to_tree_dist_program_input(write_all=True,random=True)

# path_to_output_file = "../gtp_170317/outputs/random_output.txt"
# good_dist, bad_dist = analyze_tree_dists(path_to_output_file)
#print("avg intra_good_tree_dist:{}, avg good_to_bad_tree_dist: {}".format(good_dist, bad_dist))

test_newick_str = "((((((1:12.7972272028,8:16.19448754):0.0003516433,2:13.9056052921):0.0004681324,3:9.9702006682,4:13.1786666231,6:15.5331370117):0.000904761,(5:15.0930987257,7:10.6461528477):0.000280585):0.0016585607,9:19.9632479158):0.0048466613,0:6.2155927179)"
test_tree = newick_to_tree(test_newick_str)
#check_tree(test_tree)
#print("----------------------------")
test_binary_tree = convert_tree_to_binary_rec(test_tree)
test_binary_tree = add_pred_info_to_tree(test_binary_tree, X=X, y=y)
tree_acc = test_binary_tree.eval_tree(X[-50:], y[-50:])
print("tree_acc: {}".format(tree_acc))

#check_tree(test_binary_tree)


