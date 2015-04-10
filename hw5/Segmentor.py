import numpy as np
class Segmentor():
    def __init__(self):
        self.splitRule = None
        self.left = None
        self.right = None

    def split(self, data, labels, impurity):
        # change splits for different splits
        splits = np.mean(data, 0)
        bestRule, bestFeature, bestLeft, bestRight, bestError = None, None, None, None, 1.0
        for feature in range(data.shape[1]):
            threshold = splits[feature]
            leftIndex, rightIndex = [], []
            leftHist, rightHist = { 0 : 0, 1 : 0 }, { 0 : 0, 1 : 0 }
            for i, (X, Y) in enumerate(zip(data, labels)):
                if X[feature] < threshold:
                    leftIndex.append(i)
                    leftHist[Y] += 1
                else:
                    rightIndex.append(i)
                    rightHist[Y] += 1
            error = impurity(leftHist, rightHist)
            if error < bestError:
                bestFeature = feature
                bestError = error
                bestRule = (bestFeature, threshold)
                bestLeft = leftIndex
                bestRight = rightIndex
        self.splitRule = bestRule
        self.left = (data[bestLeft], labels[bestLeft])
        self.right = (data[bestRight], labels[bestRight])
    def getSplitRule(self):
        return self.splitRule
    def getLeft(self):
        return self.left
    def getRight(self):
        return self.right




