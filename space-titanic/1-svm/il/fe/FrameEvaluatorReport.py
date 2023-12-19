class FrameEvaluatorReport:
    def __init__(self, index, slice, scores, train_accuracy, test_accuracy):
        self.index = index
        self.slice = slice
        self.scores = scores
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy

    def print(self):
        print(f'[{self.index + 1}] - With CV score of: {self.scores.mean()} and Test/Train accuracy of {self.train_accuracy} / {self.test_accuracy}')
