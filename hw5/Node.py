class AbstractNode(self):
    def traverse(self):
        raise NotImplementedError("Should never be called")

class Node(AbstractNode):
    def __init__(self, feature, threshold, left, right):
        self.split_rule = (feature, threshold)
        self.left = left
        self.right = right
    def traverse(self, X):
        if X[self.split_rule[0]] < self.split_rule[1]:
            return self.left.traverse(X)
        else:
            return self.right.traverse(X)

class LeafNode(AbstractNode):
    def __init__(self, label):
        self.Y = label

    def traverse(self, X):
        return self.Y
