import torch

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_test = X_test.size(0)
        # Compute pairwise distances using Euclidean distance
        dists = torch.cdist(X_test, self.X_train)
        # Get indices of the k nearest neighbors
        _, indices = dists.topk(self.k, largest=False, dim=1)

        predictions = torch.zeros(num_test, dtype=torch.long)
        for i in range(num_test):
            # Gather the labels of the k nearest neighbors
            nearest_classes = self.y_train[indices[i]]
            # Count the occurrences of each class in the nearest k neighbors
            votes = torch.bincount(nearest_classes, minlength=self.y_train.max() + 1)
            # Select the class with the most votes
            predictions[i] = votes.argmax()
        return predictions
