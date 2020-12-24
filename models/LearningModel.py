import os
from datetime import datetime
import joblib

class LearningModel:
    name = "Unknown Learning Model"
    model = None
    accuracy = []

    def __init__(self, name):
        self.name = name

    def fit(self, x, y):
        raise NotImplementedError("Subclass must implement fit(x,y)")

    def predict(self, x):
        return self.model.predict(x)

    def get_accuracy(self):
        return self.accuracy

    def get_model(self):
        return self.model

    def save(self, dir="saved_models"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        model = self.get_model()
        joblib.dump(model, '{}/{}.model'.format(dir, self.name.lower()).replace(' ', '_'))

    def load(self, dir="saved_models"):
        path = '{}/{}.model'.format(dir, self.name.lower()).replace(' ', '_')
        if os.path.exists(path):
            self.model = joblib.load(path)
            accuracy = []
            return True
        else:
            return False

    def __str__(self):
        return self.name
