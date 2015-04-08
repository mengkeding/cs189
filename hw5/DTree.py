import math
import random
from Node import *
import numpy as np

class DTree:
    def __init__(self, depth, impurity, segmentor):
        self.depth = depth
        self.impurity = impurity
        self.segmentor = segmentor

    #TODO: implement train and predict
    def train(self, data, labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def growTree(self, S, depth):
        # Collect Labels at maximum depth
        if depth == self.depth:
            return LeafNode(map(lambda tup: tup[1], S))

        # If all y aren't 1
        if len(filter(lambda tup: tup[1] == 1), S) == 0:
            return LeafNode([0])
        if len(filter(lambda tup: tup[1] == 0), S) == 0:
            return LeafNode([1])

        split_rule = chooseBestAttribute(self, S)
        feature = split_rule[0]
        threshold = split_rule[1]

        S_0 = []
        S_1 = []
        for tup in S:
            X = tup[0]
            if X[feature] < threshold:
                S_0.append(tup)
            else:
                S_1.append(tup)
        return Node(feature, threshold, growTree(S_0, depth+1), growTree(S_1, depth+1))


    def chooseBestAttribute(self, S):
        raise NotImplementedError

    def calculateEntropy(self, S):
        ones = sum([x[1] for x in S])
        p_1 = float(ones) / len(S)
        p_0 = 1.0 - p_1
        if p_0 == 0:
            H0 = 0
        else:
            H0 = - p_0 * math.log(p_0, 2)

        if p_1 == 0:
            H1 == 0
        else:
            H1 = - p_1 * math.log(p_1, 2)
        return H0 + H1
