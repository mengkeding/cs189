from Node import *
import numpy as np
import random

class DTree:
    def __init__(self, impurity, segmentor, depth=50, randomness=None, name=None):
        self.impurity = impurity
        self.segmentor = segmentor
        self.depth = depth
        self.randomness = randomness
        self.name = name
        self.root = None

    def train(self, data, labels):
        if self.randomness:
            features = list(range(data.shape[1]))
            random.shuffle(features)
            features = features[:self.randomness]
        else:
            features = None

        self.segmentor.split(data, labels, self.impurity, features)
        (leftData, leftLabels) = self.segmentor.getLeft()
        (rightData, rightLabels) = self.segmentor.getRight()
        splitRule = self.segmentor.getSplitRule()
        def growTree(data, labels):
            S = np.unique(labels)
            # No data
            if len(S) == 0:
                return Node()
            # Create a leaf since only one label
            elif len(S) == 1:
                return Node(label=S[0], distribution={S[0] : 1})
            else:
                if self.depth == 0:
                    numLabels = np.bincount(S)
                    tmp = numLabels / float(sum(numLabels))
                    d = dict([(i, tmp[i]) for i in range(len(tmp))])
                    return Node(label=np.argmax(numLabels), distribution=d)
                else:
                    return DTree(self.impurity, self.segmentor, depth=self.depth-1, randomness=self.randomness).train(data, labels)
        left = growTree(leftData, leftLabels)
        right = growTree(rightData, rightLabels)
        feature = splitRule[0]
        threshold = splitRule[1]
        self.root = Node(feature, threshold, left, right)
        return self.root

    def predict(self, data, single=False):
        def traverse(S):
            node = self.root
            while not node.isLeaf():
                feature = node.getFeature()
                threshold = node.getThreshold()
                if S[feature] < threshold:
                    node = node.left
                else:
                    node = node.right
            return node.distribution if single else node.label
        if single:
            return traverse(data)
        else:
            return np.array(map(traverse, data))
