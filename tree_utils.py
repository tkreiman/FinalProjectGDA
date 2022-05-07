import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from numpy import random
import matplotlib.pyplot as plt
import scipy

from sklearn.datasets import load_digits

class TreeNode:
    def __init__(self, leafID, children, lengths, binary=False):
        self.leafID = leafID
        self.children = children
        self.lengths = lengths
        self.binary = binary
        self.threshold = None
        self.feature = None
        self.idx = None
        if self.children is not None or self.lengths is not None:
            assert(len(self.children) == len(self.lengths))
    def get_num_nodes(self):
        num_nodes = 1
        if self.children:
            for i in range(len(self.children)):
                num_nodes += self.children[i].get_num_nodes()
        return num_nodes
    
    def get_leaves(self, leaf_nums):
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
            return self.children[0].get_leaves([])
        else:
            raise Exception("leaf node has no children")

    def get_right_leaves(self):
        if not self.binary:
            raise Exception("tree not binary!")
        if self.children:
            return self.children[1].get_leaves([])
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

    def _add_nodes_to_list(self, node_list):
        node_list.append(self)
        if self.children:
            self.children[0]._add_nodes_to_list(node_list)
            self.children[1]._add_nodes_to_list(node_list)


    def label_tree(self):
        node_list = []
        self._add_nodes_to_list(node_list)
        assert(len(node_list) == self.get_num_nodes())
        for i in range(len(node_list)):
            node_list[i].idx = i
    
    def _get_tree_array_info(self, left_children, right_children, lengths, leaf_class_mapping):
        if self.children:
            left_children[self.idx] = self.children[0].idx
            right_children[self.idx] = self.children[1].idx
            lengths[self.children[0].idx] = self.lengths[0]
            lengths[self.children[1].idx] = self.lengths[1]
            self.children[0]._get_tree_array_info(left_children, right_children, lengths, leaf_class_mapping)
            self.children[1]._get_tree_array_info(left_children, right_children, lengths, leaf_class_mapping)
        else:
            left_children[self.idx] = -1
            right_children[self.idx] = -1
            leaf_class_mapping[self.idx] = self.leafID #class info
        
    def get_tree_array_info(self):
        self.label_tree()
        num_nodes = self.get_num_nodes()
        left_children = np.zeros(num_nodes, dtype=int)
        right_children = np.zeros(num_nodes, dtype=int)
        lengths = np.zeros(num_nodes, dtype=float)
        leaf_class_mapping = {}
        self._get_tree_array_info(left_children, right_children, lengths, leaf_class_mapping)
        return left_children, right_children, lengths, leaf_class_mapping
    
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
def build_newick_from_dt(clf, add_leaf_class_edges = False, match_class_leaves = False, mode="sklearn"):
    global curr_leaf_num
    if add_leaf_class_edges:
        assert(mode=="sklearn")
    curr_leaf_num = 0
    if mode == "sklearn":
        children_left = clf.tree_.children_left #children_left[i] gives the id of the left child of node i
        children_right = clf.tree_.children_right
        if match_class_leaves:
            leaf_node_ids = [i for i in range(len(clf.tree_.children_left)) if clf.tree_.children_left[i] == -1]
            def leaf_purity_sort_func(e):
                return clf.tree_.impurity[e]
            leaf_node_ids.sort(key=leaf_purity_sort_func)
            leaf_class_mapping = {}
            total_retries = 0
            for leaf_id in leaf_node_ids:
                for _ in range(len(clf.tree_.value[leaf_id][0])):
                    leaf_class = np.argmax(clf.tree_.value[leaf_id][0])
                    if leaf_class not in leaf_class_mapping.values():
                        leaf_class_mapping[leaf_id] = leaf_class
                        break
                    else:
                        total_retries += 1
                        clf.tree_.value[leaf_id][0][leaf_class] = 0
                        continue
            #print("total class assignment retries: {}".format(total_retries))
            bad_leaf_count = 0
            for leaf_id in leaf_node_ids:
                if leaf_id not in leaf_class_mapping.keys():
                    bad_leaf_count += 1
                    for i in range(len(clf.classes_)):
                        if i not in leaf_class_mapping.values():
                            leaf_class_mapping[leaf_id] = i
            #print("this decision tree has {} bad leaves".format(bad_leaf_count))
            lengths = calc_edge_lengths_from_dt(clf) #lengths[i] gives the length of the edge to node i from parent
    else:
        assert(isinstance(clf, TreeNode))
        children_left, children_right, lengths, leaf_class_mapping = clf.get_tree_array_info()

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

def test_tree_conversion_pipeline(bootstrap_number):
    indices = bootstrap_sample(bootstrap_number,NUM_TRAIN,len(X))
    percent_diffs = np.zeros(bootstrap_number)
    for i in range(bootstrap_number):
        Xtr = X[indices[i][0]]
        ytr = y[indices[i][0]]
        Xtst = X[indices[i][1]]
        ytst = y[indices[i][1]]
        dt = tree.DecisionTreeClassifier(max_leaf_nodes=10)
        dt = dt.fit(Xtr,ytr)
        nw = build_newick_from_dt(dt, match_class_leaves=True)
        og_score = dt.score(Xtst,ytst)
        coverted_dt = newick_to_treenode(nw, Xtr, ytr)
        new_score = coverted_dt.eval_tree(Xtst,ytst)
        percent_diff = (new_score - og_score)/og_score
        print("original score: {}, score after conversion: {}, percent_diff: {}".format(og_score, new_score, percent_diff))
        percent_diffs[i] = percent_diff
        print("--------------------------")
    print("average percent accuracy diff due to conversion: {}".format(np.mean(percent_diffs)))


def get_data_class_rep(X, y, classes):
    class_rep_info = np.zeros((len(classes), X.shape[1]), dtype=X.dtype)
    for i, c in enumerate(classes):
        idxs = np.where(y == c, True, False)
        X_cls = X[idxs]
        class_rep_info[i] = np.mean(X_cls, axis=0)
    return class_rep_info

def find_matching_parens_idx(string):
    stack_height = 1
    for i in range(len(string)):
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
    while i < len(newick_str):
        if newick_str[i] == "(":
            end_idx = find_matching_parens_idx(newick_str[i+1:])
            childNode = newick_to_tree_rec(newick_str[i+1:i+1+end_idx])
            children.append(childNode)
            length, next_idx = get_len_and_next_idx(newick_str, i+1+end_idx + 2)
            lengths.append(length)
            i = next_idx
        else:
            leafID = int(newick_str[i]) #doesn't generalize to multi-digit labels
            leafNode = TreeNode(leafID=leafID,children=None,lengths=None)
            children.append(leafNode)
            length, next_idx = get_len_and_next_idx(newick_str, i + 2)
            lengths.append(length)
            i = next_idx
    return TreeNode(leafID=-1, children=children, lengths=lengths)

def newick_to_tree(newick_str):
    if newick_str[-1] == ";":
        newick_str = newick_str[:-1]
    print("newick str: {}".format(newick_str))
    return newick_to_tree_rec(newick_str[1:-1])

def newick_to_treenode(newick_str, Xtr, ytr):
    tree = newick_to_tree(newick_str)
    print("original number of nodes:{}".format(tree.get_num_nodes()))
    tree = convert_tree_to_binary_rec(tree)
    print("binary number of nodes:{}".format(tree.get_num_nodes()))
    tree = add_pred_info_to_tree(tree, X=Xtr, y=ytr)
    print("processed binary number of nodes:{}".format(tree.get_num_nodes()))
    return tree

def check_tree(node:TreeNode, level=0):
    if node.children is None:
        num_children = 0
    else:
        num_children = len(node.children)
        print("curr children:{}".format(node.children))
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
        new_left_node = convert_tree_to_binary_rec(node.children[0])
        new_right_node = TreeNode(leafID=-1, children=node.children[1:], lengths=node.lengths[1:])
        new_right_node = convert_tree_to_binary_rec(new_right_node)
        new_children = [new_left_node, new_right_node]
        #total_lengths = node.lengths[0] = sum(new_binary_node.lengths)
        #length1 = node.lengths[0]/ 
        #TODO: might come up with better length reweighing scheme
        new_lengths = [node.lengths[0], np.mean(node.lengths[1:])]
        return TreeNode(leafID=-1, children=new_children, lengths=new_lengths, binary=True)

def choose_feature(X, y, left_leaves, right_leaves, mode="max_var", class_reps=None, theshold=None):
    if mode == "max_var":
        assert(class_reps is not None)
        left_reps = np.mean(class_reps[left_leaves], axis=0)
        right_reps = np.mean(class_reps[right_leaves], axis=0)
        feat_diff = np.abs(left_reps - right_reps)
        max_diff_feat = np.argmax(feat_diff)
        return max_diff_feat
    elif mode == "entropy":
        left_idxs = [i for i in range(len(X)) if y[i] in left_leaves]
        right_idxs = [i for i in range(len(X)) if y[i] in right_leaves]
        min_idx_len = min(len(left_idxs), len(right_idxs))
        left_idxs = left_idxs[:min_idx_len]
        right_idxs = right_idxs[:min_idx_len]
        left_data = X[left_idxs]
        right_data = X[right_idxs]
        feat_entropy_info = scipy.stats.entropy(pk=left_data, qk=right_data, axis=0)
        min_entropy_feat = np.argmin(feat_entropy_info)
        return min_entropy_feat
    elif mode == "pred_potential":
        assert (theshold is not None)
        feat_correct = np.zeros(X.shape[1])
        for f in range(X.shape[1]):
            for i in range(X.shape[0]):
                if X[i][f] < theshold:
                    if y[i] in left_leaves:
                        feat_correct[f] += 1
                else:
                    if y[i] in right_leaves:
                        feat_correct[f] += 1
        return np.argmax(feat_correct)        
    else:
        raise ValueError("Mode not supported")

def process_feat_info(node, X, y, class_reps):
    assert(node.binary)
    if node.children is None:
        return
    else:
        #thresh = DATA_MIN + (DATA_MAX - DATA_MIN) * (1 - (node.lengths[0]/node.lengths[1]))
        #thresh = DATA_MIN + node.lengths[0]
        thresh = ((DATA_MIN + node.lengths[0]) +  (DATA_MAX - node.lengths[1]))/2
        node.threshold = thresh
        left_leaves = node.get_left_leaves()
        right_leaves = node.get_right_leaves()
        #get feature that varies the most between left classes and right classes
        #node.feature = np.random.randint(len(feat_diff))
        node.feature = choose_feature(X, y, left_leaves, right_leaves, mode="pred_potential", class_reps=class_reps, theshold=thresh)
        process_feat_info(node.children[0], X, y, class_reps)
        process_feat_info(node.children[1], X, y, class_reps)
        return
    
def add_pred_info_to_tree(node, X, y):
    classes = np.arange(10)
    class_reps = get_data_class_rep(X=X, y=y, classes=classes)
    process_feat_info(node, X, y, class_reps)
    return node
    
def write_to_tree_dist_program_input(mode="outliers", random=False):
    #mode = "all", "outliers", "good", or "bad"
    forest_test = find_good_trees(500, random=random)
    nws, trees, scores = zip(*forest_test)
    scores = np.array(scores)
    print("average score: {}".format(np.mean(scores)))
    sorted_idxs = np.argsort(scores)
    bad_idxs = sorted_idxs[:5]
    good_idxs = sorted_idxs[-5:]
    print("bad scores:{}".format(scores[bad_idxs]))
    print("good scores:{}".format(scores[good_idxs]))
    print("average good score: {}".format(np.mean(scores[good_idxs])))
    with open("../gtp_170317/example/final_5_best", "w") as f:
        if mode == "all":
            for i in range(len(nws)):
                f.write(nws[i])
                f.write('\n')
        else:
            if mode == "outliers" or mode == "good":
                for i in range(len(good_idxs)):
                    f.write(nws[good_idxs[i]])
                    f.write('\n')
            if mode == "outliers" or mode == "bad":
                for i in range(len(bad_idxs)):
                    f.write(nws[bad_idxs[i]])
                    f.write('\n')
  
def analyze_tree_dists(filename, all_dists=False):
    dist_matrix = np.zeros((10, 10))
    with open(filename, "r") as f:
        #num_lines = len(f.readlines())
        for line in f:
            if line == "\n":
                continue
            line = line.split("\t")
            t1 = int(line[0])
            t2 = int(line[1])
            dist = float(line[2])
            dist_matrix[t1][t2] = dist
            dist_matrix[t2][t1] = dist
    #print(dist_matrix)
    if all_dists:
        tot_dist = 0
        for i in range(10):
            for j in range(10):
                tot_dist += dist_matrix[i][j]
        tot_dist /= 90
        print("total distance between trees: {}".format(tot_dist))  
    else:
        #print("distances to good mean tree from random trees: {}".format(dist_matrix[0]))
        intra_good_tree_dist = 0
        good_to_bad_tree_dist = 0
        for i in range(5):
            for j in range(5):
                intra_good_tree_dist += dist_matrix[i][j]
            for j in range(5, 10):
                good_to_bad_tree_dist += dist_matrix[i][j]
        intra_bad_tree_dist = 0
        for i in range(5, 10):
            for j in range(5, 10):
                intra_bad_tree_dist += dist_matrix[i][j]

        intra_good_tree_dist /= 20
        intra_bad_tree_dist /= 20
        good_to_bad_tree_dist /= 25
        print("avg intra_good_tree_dist:{}, avg good_to_bad_tree_dist: {}, avg intra_bad_tree_dist:{}".format(intra_good_tree_dist, good_to_bad_tree_dist, intra_bad_tree_dist))
    
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
# data_std = np.sqrt(np.mean(np.var(X, axis=0)))
# print("data mean: {}, data std: {}".format(np.mean(X), data_std))
test_X = X[-100:]
test_y = y[-100:]
X = X[:-100]
y = y[:-100]

NUM_TRAIN = int(len(X) * 0.5)

# path_to_output_file = "../gtp_170317/outputs/random_10"
# analyze_tree_dists(path_to_output_file, all_dists=True)

#write_to_tree_dist_program_input(mode="good",random=False)


#MEAN TREE FORMAT EXPERIMENTS BELOW:
mean_newick = "(((((((1:8.0217932068,8:20.3863713218):7.8439513396,2:12.8791334299):7.9591532337,4:16.432357059):5.9429434874,7:6.69057821):5.7858385669,(3:7.9174358111,9:21.0825641889):7.8720193078):6.8763499013,0:3.6980332976):15.6004908374,(5:16.2160152503,6:13.9161482085):3.8111437655)"
indices = bootstrap_sample(1, NUM_TRAIN, len(X))
Xtr = X[indices[0][0]]
ytr = y[indices[0][0]]
mean_tree = newick_to_treenode(mean_newick, Xtr, ytr)
tree_acc = mean_tree.eval_tree(test_X, test_y)
print("tree_acc: {}".format(tree_acc))

# test_binary_tree.label_tree()
# test_mean_newick = build_newick_from_dt(test_binary_tree, match_class_leaves=True, mode="treenode")
# print("newick_format: {}".format(test_mean_newick))

#CONVERSION TEST PERFORMANCE
#test_tree_conversion_pipeline(50)

