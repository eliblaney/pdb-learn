import os
from datetime import datetime
import joblib
import uuid

class LearningModel:
    name = "Unknown Learning Model"
    options = None
    model = None
    accuracy = []
    mutator = None
    uuid = None

    def __init__(self, name, options):
        self.name = name
        self.options = options
        self.uuid = str(uuid.uuid4())

    def fit(self, x, y):
        raise NotImplementedError("Subclass must implement fit(x,y)")

    def partial_fit(self, x, y):
        raise NotImplementedError("Subclass must implement partial_fit(x,y)")

    def predict(self, x):
        return self.model.predict(x)

    def get_accuracy(self):
        return self.accuracy

    def get_model(self):
        return self.model

    def get_options(self):
        return self.options

    def save(self, dir="saved_models"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        model = self.get_model()
        joblib.dump(model, '{}/{}.model'.format(dir, self.get_full_name().lower().replace(' ', '_'), uuid))

    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.accuracy = []
            return True
        else:
            return False

    @staticmethod
    def get_default_options():
        raise NotImplementedError("Subclass must implement get_default_options()")

    def __str__(self):
        return self.name

    def get_full_name(self):
        return self.name + '.' + self.uuid
