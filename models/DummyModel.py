import random
from models.LearningModel import LearningModel

class DummyModel(LearningModel):

    def __init__(self, options):
        super().__init__("Logistic Regression", options)
        print(options)

    @staticmethod
    def get_default_options():
        return {'I':['a','b','c'],'II':['1','2','3']}

    def get_accuracy(self):
        return ord(self.options['I']) - ord(self.options['II'])

    def __str__(self):
        return "DummyModel: " + str(self.get_options())

