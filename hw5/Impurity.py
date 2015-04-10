import numpy as np
import math

class Impurity():
    @staticmethod
    def impurity(left, right):
        def entropy(S):
            total = float(S[0] + S[1])
            p_0 = (S[0] + 1e-5) / (total + 1e-5)
            p_1 = (S[1] + 1e-5) / (total + 1e-5)
            H = (-p_0 * math.log(p_0, 2)) + (-p_1 * math.log(p_1, 2))
            return H
        numLeft = left[0] + left[1]
        numRight = right[0] + right[1]
        total = numLeft + numRight
        wLeft = float(numLeft) / total
        wRight = float(numRight) / total
        return wLeft*entropy(left) + wRight*entropy(right)

