import math
import random
from Node import *
import numpy as np
from collections import Counter

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
        S_0 = filter(lambda tup: tup[0] == 0, S)
        S_1 = filter(lambda tup: tup[0] == 1, S)
        counter0 = Counter(map(lambda tup: tup[1], S_0))
        counter1 = Counter(map(lambda tup: tup[1], S_0))
        y_0 = counter0.most_common(1)[0][0]
        y_1 = counter1.most_common(1)[0][0]
        J_0 = filter(lambda tup: tup[0] == 0, counter0)[0][1]
        J_1 = filter(lambda tup: tup[0] == 1, counter0)[0][1]
        total_error = J_0 + J_1



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
