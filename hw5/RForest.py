from DTree import *
import operator
from collections import defaultdict

class RForest():
    def __init__(self, impurity, segmentor, depth=10, randomness=None, nTrees=10, samples=None):
        """
        Constructor
        impurity - described in spec
        segmentor - described in spec
        depth - max depth
        randomness - num features
        nTrees = number of trees
        samples - number of data points
        """
        self.impurity = impurity
        self.segmentor = segmentor
        self.depth = depth
        self.randomness = randomness
        self.nTrees = nTrees
        self.samples = samples
        self.forest = []

    def train(self, data, labels):
        def sample(data, labels):
            n = data.shape[0]
            samples = self.samples if self.samples else (n/2)
            i = np.random.choice(n, samples, replace=False)
            return data[i], labels[i]
        for index in range(self.nTrees):
            tree = DTree(self.impurity, self.segmentor, self.depth, self.randomness, name=index)
            sData, sLabel = sample(data, labels)
            tree.train(sData, sLabel)
            self.forest.append(tree)

    def predict(self, data):
        def traverse(data):
            distributions = []
            for tree in self.forest:
                distribution = tree.predict(data, single=True)
                distributions.append(distribution)
            count = defaultdict(int)
            for d in distributions:
                for label in d:
                    count[label] += d[label]
            # Returns most frequent label
            return max(count.iteritems(), key=operator.itemgetter(1))[0]
        return np.array(map(traverse, data))
