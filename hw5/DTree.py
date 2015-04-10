from Node import *
import numpy as np

class DTree:
    def __init__(self, impurity, segmentor, depth=50):
        self.impurity = impurity
        self.segmentor = segmentor
        self.depth = depth
        self.root = None

    def train(self, data, labels):
        self.segmentor.split(data, labels, self.impurity)
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
                return Node(label=S[0])
            else:
                if self.depth == 0:
                    numLabels = np.bincount(S)
                    return Node(label=np.argmax(numLabels))
                else:
                    return DTree(self.impurity, self.segmentor, depth=self.depth-1).train(data, labels)
        left = growTree(leftData, leftLabels)
        right = growTree(rightData, rightLabels)
        feature = splitRule[0]
        threshold = splitRule[1]
        self.root = Node(feature, threshold, left, right)
        return self.root

    def predict(self, data):
        def traverse(S):
            node = self.root
            while not node.isLeaf():
                feature = node.getFeature()
                threshold = node.getThreshold()
                if S[feature] < threshold:
                    node = node.left
                else:
                    node = node.right
            return node.label
        return np.array(map(traverse, data))
