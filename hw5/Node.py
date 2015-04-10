class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None, distribution=None):
        self.split_rule = (feature, threshold)
        self.left = left
        self.right = right
        # leaf nodes only
        self.label=label
        self.distribution = distribution

    def getFeature(self):
        return self.split_rule[0]

    def getThreshold(self):
        return self.split_rule[1]

    def isLeaf(self):
        return not self.left and not self.right
