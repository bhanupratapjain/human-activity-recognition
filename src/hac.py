import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import __root__

plt.style.use('ggplot')
# %matplotlib inline


from sklearn import decomposition
from sklearn import datasets

PROJECT_ROOT = __root__.path()


class Data:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, train_rows=None, test_rows=None):
        data_folder = os.path.join(PROJECT_ROOT, "data")
        X_train_file_path = os.path.join(data_folder, "Train/X_train.txt")
        y_train_file_path = os.path.join(data_folder, "Train/y_train.txt")
        X_test_file_path = os.path.join(data_folder, "Test/X_test.txt")
        y_test_file_path = os.path.join(data_folder, "Test/y_test.txt")
        self.X_train = np.loadtxt(X_train_file_path, delimiter=' ')[0:train_rows, :]
        self.y_train = np.loadtxt(y_train_file_path, delimiter=' ')[0:train_rows, ]
        self.X_test = np.loadtxt(X_test_file_path, delimiter=' ')[0:test_rows, :]
        self.y_test = np.loadtxt(y_test_file_path, delimiter=' ')[0:test_rows, ]
        # self.y_train = np.loadtxt(y_test_file_path, delimiter=' ').reshape(-1, 1)
        # self.y_test = np.loadtxt(y_test_file_path, delimiter=' ').reshape(-1, 1)


class Classifiers:
    def __init__(self, data):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.y_predict = {}  # {k - name_of_classifier, v - [predictions]}
        self.c_list = {}  # {k - name_of_classifier, v - sklearn_classifier_object}
        self.scores = {}  # {k - name_of_classifier, { k - score_metric, v - score}}

    def add_classifier(self, classifier_name, classifier):
        self.c_list[classifier_name] = classifier

    def fit(self):
        for classifier_name, classifier in self.c_list.iteritems():
            classifier.fit(self.X_train, self.y_train)

    def predict(self):
        for classifier_name, classifier in self.c_list.iteritems():
            self.y_predict[classifier_name] = classifier.predict(self.X_test)

    def get_scores(self):
        for classifier_name, classifier in self.c_list.iteritems():
            self.__get_score(classifier_name, classifier)

    def __get_score(self, classifier_name, classifier):
        self.scores[classifier_name] = {
            "accuracy": accuracy_score(self.y_test, self.y_predict[classifier_name])
        }

    def print_scores(self):
        pprint.pprint(self.scores)


class FeatureSelection:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def pca(self):
        pass

    def pca_example(self):
        np.random.seed(5)

        centers = [[1, 1], [-1, -1], [1, -1]]
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)

        for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
            ax.text3D(X[y == label, 0].mean(),
                      X[y == label, 1].mean() + 1.5,
                      X[y == label, 2].mean(), name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.show()


def test1():
    data = Data()
    data.load_data()
    print data.y_train.shape
    # fs = FeatureSelection()
    # fs.pca_example()
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(data.X_train, data.y_train)

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_pos = rfc.predict_proba(data.X_test)
    print rfc.score(data.X_test, data.y_test)
    print prob_pos[0]
    print len(prob_pos)
    pprint.pprint(prob_pos)
    # prob_pos = \
    #     (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(data.y_test, prob_pos, n_bins=12)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % ("rf",))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label="rf",
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test1()
    # a  = np.arange(15).reshape(5,3)
    # print a[0:None,:]

    data = Data()
    data.load_data()
    # print data.X_train
    print data.X_train.shape
    cls = Classifiers(data)
    rfc = RandomForestClassifier(n_estimators=100)
    cls.add_classifier("rfc", rfc)
    cls.fit()
    cls.predict()
    cls.get_scores()
    cls.print_scores()
