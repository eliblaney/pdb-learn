from sklearn import ensemble
from models.LearningModel import LearningModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class RandomForestClassifier(LearningModel):
    def __init__(self, n_splits=3):
        super().__init__("Random Forest Classifier")
        self.model = ensemble.RandomForestClassifier(n_estimators=10)
        self.kf = StratifiedKFold(n_splits, shuffle=False)

    def fit(self, x, y):
        for train_index, test_index in self.kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train)
            self.accuracy.append(accuracy_score(y_test, self.model.predict(x_test), normalize=True)*100)
