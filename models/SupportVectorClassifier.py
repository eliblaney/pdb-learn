from sklearn import svm
from models.LearningModel import LearningModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class SupportVectorClassifier(LearningModel):
    def __init__(self, n_splits=5, kernel=2, degree=3, coef0=0.0, probability=False):
        super().__init__("Support Vector Classifier")
        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        self.model = svm.SVC(kernel=kernels[kernel], degree=degree, coef0=coef0, probability=probability)
        self.kf = StratifiedKFold(n_splits, shuffle=False)

    def fit(self, x, y):
        for train_index, test_index in self.kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train)
            self.accuracy.append(accuracy_score(y_test, self.model.predict(x_test), normalize=True)*100)
