from sklearn import linear_model
from models.LearningModel import LearningModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import explained_variance_score

class LogisticRegression(LearningModel):
    def __init__(self, options, n_splits=3):
        super().__init__("Logistic Regression", options)
        if options is not None:
            self.model = linear_model.LogisticRegression(solver=options['solver'])
        self.kf = StratifiedKFold(n_splits, shuffle=False)

    def fit(self, x, y):
        for train_index, test_index in self.kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train)
            self.accuracy.append(explained_variance_score(y_test, self.model.predict(x_test)))

    @staticmethod
    def get_default_options():
        return {
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
