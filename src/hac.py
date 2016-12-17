from __future__ import print_function

import os
import pprint
import threading
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

import __root__

plt.style.use('ggplot')
# %matplotlib inline


PROJECT_ROOT = __root__.path()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class Data:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.labels = None

    def load_data(self, train_rows=None, test_rows=None, shfl=False, scl=True):
        data_folder = os.path.join(PROJECT_ROOT, "data")
        X_train_file_path = os.path.join(data_folder, "Train/X_train.txt")
        y_train_file_path = os.path.join(data_folder, "Train/y_train.txt")
        X_test_file_path = os.path.join(data_folder, "Test/X_test.txt")
        y_test_file_path = os.path.join(data_folder, "Test/y_test.txt")
        labels_file_path = os.path.join(data_folder, "activity_labels.txt")
        self.X_train = np.loadtxt(X_train_file_path, delimiter=' ')[
                       0:train_rows, :]
        self.y_train = np.loadtxt(y_train_file_path, delimiter=' ')[
                       0:train_rows, ]
        self.X_test = np.loadtxt(X_test_file_path, delimiter=' ')[0:test_rows,
                      :]
        self.y_test = np.loadtxt(y_test_file_path, delimiter=' ')[0:test_rows, ]
        lb = np.genfromtxt(labels_file_path, delimiter=' ', dtype=None)
        self.labels = np.asarray(map(lambda x: x[1], lb))
        # self.labels = np.asarray(np.genfromtxt(labels_file_path,
        # delimiter=' ',dtype=None))[:,-1]

        # self.y_train = np.loadtxt(y_test_file_path, delimiter=' ').reshape(
        # -1, 1)
        # self.y_test = np.loadtxt(y_test_file_path, delimiter=' ').reshape(
        # -1, 1)
        if scl:
            StandardScaler().fit_transform(self.X_train)
        if shfl:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)


class Classifiers:
    def __init__(self, data):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        self.y_predict = {}  # {k - name_of_classifier, v - [predictions]}
        self.cls = {}  # {k - name_of_classifier, v - sklearn_classifier_object}
        self.scores = {}  # {k - name_of_classifier, { k - score_metric,
        # v - score}}

    def add_classifier(self, classifier_name, classifier):
        self.cls[classifier_name] = classifier

    def __fit(self, cls, cls_name, X, y):
        t0 = time()
        cls.fit(X, y)
        print("%s done in %0.3fs" % (cls_name, time() - t0))

    def fit(self):
        threads = []
        for classifier_name, classifier in self.cls.iteritems():
            th = threading.Thread(target=self.__fit, args=(
                classifier, classifier_name, self.X_train, self.y_train,))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

    def predict(self):
        for classifier_name, classifier in self.cls.iteritems():
            self.y_predict[classifier_name] = classifier.predict(self.X_test)

    def get_scores(self):
        for classifier_name, classifier in self.cls.iteritems():
            self.__get_score(classifier_name, classifier)

    def __get_score(self, classifier_name, classifier):
        self.scores[classifier_name] = {
            "accuracy": accuracy_score(self.y_test,
                                       self.y_predict[classifier_name]),
            "precision": precision_score(self.y_test,
                                         self.y_predict[classifier_name],
                                         average='weighted'),
            "recall": recall_score(self.y_test,
                                   self.y_predict[classifier_name],
                                   average='weighted')
            # "confusion_matrix": confusion_matrix(self.y_test,
            # self.y_predict[classifier_name])
        }

    def print_scores_sys(self):
        pprint.pprint(self.scores)

    def print_scores(self):
        df = pd.DataFrame(self.scores)
        display(df)

    def show_plt(self):
        cls_names = []
        accuracy = []
        precision = []
        recall = []
        for name, score_values in self.scores.iteritems():
            cls_names.append(name)
            # print(score_values)
            for score_type, v in score_values.iteritems():
                if score_type == 'accuracy':
                    accuracy.append(v)
                elif score_type == 'precision':
                    precision.append(v)
                elif score_type == 'recall':
                    recall.append(v)
        accuracy, cls_names = (list(t) for t in
                               zip(*sorted(zip(accuracy, cls_names))))
        plt.figure(figsize=(7, 7))
        plt.barh(np.arange(len(cls_names)), accuracy, align='center', alpha=0.4)
        plt.yticks(np.arange(len(cls_names)), cls_names)
        plt.ylabel("Classifiers")
        plt.xlabel("Accuracy")
        plt.subplots_adjust(left=.3)
        plt.title("Accuracy vs. Classifiers")


def cls_compare_no_shuff():
    data = Data()
    data.load_data(shfl=False)
    cls = Classifiers(data)
    dtc = DecisionTreeClassifier()
    gnb = GaussianNB()
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    linear_svc = svm.SVC(kernel='linear', class_weight='balanced')
    poly_svc = svm.SVC(kernel='poly', class_weight='balanced')
    rbf_svc = svm.SVC(kernel='rbf', class_weight='balanced')
    # LinearSVC minimizes the squared hinge loss while SVC minimizes the
    # regular hinge loss.
    # LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass
    # reduction while
    # SVC uses the One-vs-One multiclass reduction.
    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier(12)
    cls.add_classifier("knn", knn)
    cls.add_classifier("decision-trees", dtc)
    cls.add_classifier("random-forest", rfc)
    cls.add_classifier("gaussian-naive-bayes", gnb)
    cls.add_classifier("linear-discriminant-analysis", lda)
    cls.add_classifier("quadratic-discriminant-analysis", qda)
    cls.add_classifier("linear-support-vector-machine", linear_svc)
    cls.add_classifier("poly-support-vector-machine", poly_svc)
    cls.add_classifier("rbf-support-vector-machine", rbf_svc)
    cls.fit()
    cls.predict()
    cls.get_scores()
    cls.print_scores()
    cls.print_scores_sys()
    cls.show_plt()


def cls_compare_shuff():
    data = Data()
    data.load_data(shfl=True)
    cls = Classifiers(data)
    dtc = DecisionTreeClassifier()
    gnb = GaussianNB()
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    linear_svc = svm.SVC(kernel='linear')
    poly_svc = svm.SVC(kernel='poly')
    rbf_svc = svm.SVC(kernel='rbf')
    # LinearSVC minimizes the squared hinge loss while SVC minimizes the
    # regular hinge loss.
    # LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass
    # reduction while
    # SVC uses the One-vs-One multiclass reduction.

    rfc = RandomForestClassifier()
    knn = KNeighborsClassifier(12)
    cls.add_classifier("knn", knn)
    cls.add_classifier("decision-trees", dtc)
    cls.add_classifier("random-forest", rfc)
    cls.add_classifier("gaussian-naive-bayes", gnb)
    cls.add_classifier("linear-discriminant-analysis", lda)
    cls.add_classifier("quadratic-discriminant-analysis", qda)
    cls.add_classifier("linear-support-vector-machine", linear_svc)
    cls.add_classifier("poly-support-vector-machine", poly_svc)
    cls.add_classifier("rbf-support-vector-machine", rbf_svc)
    cls.fit()
    cls.predict()
    cls.get_scores()
    # cls.print_scores()
    cls.print_scores_sys()
    cls.show_plt()


def svm_all_unweighted():
    data = Data()
    data.load_data(shfl=True)
    cls = Classifiers(data)
    linear_svc = svm.SVC(kernel='linear')
    poly_svc = svm.SVC(kernel='poly')
    rbf_svc = svm.SVC(kernel='rbf')
    cls.add_classifier("linear-support-vector-machine", linear_svc)
    cls.add_classifier("poly-support-vector-machine", poly_svc)
    cls.add_classifier("rbf-support-vector-machine", rbf_svc)
    cls.fit()
    cls.predict()
    cls.get_scores()
    cls.print_scores()
    cls.show_plt()


def svm_all_weighted():
    data = Data()
    data.load_data(shfl=True)
    cls = Classifiers(data)
    linear_svc = svm.SVC(kernel='linear', C=1.00092594323,
                         class_weight='balanced')
    poly_svc = svm.SVC(kernel='poly', class_weight='balanced')
    rbf_svc = svm.SVC(kernel='rbf', class_weight='balanced')
    cls.add_classifier("linear-support-vector-machine", linear_svc)
    cls.add_classifier("poly-support-vector-machine", poly_svc)
    cls.add_classifier("rbf-support-vector-machine", rbf_svc)
    cls.fit()
    cls.predict()
    cls.get_scores()
    cls.print_scores()
    cls.show_plt()


def svm_linear_grid_search():
    data = Data()
    data.load_data()
    cls = Classifiers(data)
    # Default c = 1
    param_grid = {'C': [1, 1e3, 5e3, 1e4, 5e4, 1e5]}

    # linear_svc = GridSearchCV(svm.SVC(kernel='linear',
    # class_weight='balanced'),
    #                           param_grid)
    linear_svc = GridSearchCV(
        svm.SVC(kernel='linear'),
        param_grid)
    cls.add_classifier("linear-support-vector-machine", linear_svc)
    cls.fit()
    print("Best estimator found by grid search:")
    print(cls.cls['linear-support-vector-machine'].best_estimator_)
    cls.predict()
    cls.get_scores()
    cls.print_scores()
    # cls.show_plt()


def cls_compare_pca():
    pca_values = [5, 10, 25, 30, 40, 50, 100, 150, 200, 250, 300, 561]
    data = Data()
    data.load_data(shfl=True)
    scores = {}
    for v in pca_values:
        print("Running PCA with %s components" % v)
        pca = PCA(n_components=v)
        pca.fit_transform(data.X_train)
        pca.transform(data.X_test)
        cls = Classifiers(data)
        dtc = DecisionTreeClassifier()
        gnb = GaussianNB()
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()
        linear_svc = svm.SVC(kernel='linear', class_weight='balanced')
        poly_svc = svm.SVC(kernel='poly', class_weight='balanced')
        rbf_svc = svm.SVC(kernel='rbf', class_weight='balanced')
        knn = KNeighborsClassifier(12)
        cls.add_classifier("knn", knn)
        cls.add_classifier("decision-trees", dtc)
        cls.add_classifier("gaussian-naive-bayes", gnb)
        cls.add_classifier("linear-discriminant-analysis", lda)
        cls.add_classifier("quadratic-discriminant-analysis", qda)
        cls.add_classifier("linear-support-vector-machine", linear_svc)
        cls.add_classifier("poly-support-vector-machine", poly_svc)
        cls.add_classifier("rbf-support-vector-machine", rbf_svc)
        cls.fit()
        cls.predict()
        cls.get_scores()
        cls.print_scores()
        for k, v in cls.scores.items():
            try:
                scores[k] += [v["accuracy"]]
            except KeyError:
                scores[k] = [v["accuracy"]]
    for k, v in scores.items():
        plt.plot(pca_values, v, label=k)
    plt.xlabel("PCA Values")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)


def get_scores(y_test, y_pred):
    # Reads labels and predictions and gives accuracy, precision, recall &
    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    prec = np.around(np.diag(cm).astype(float) * 100 / cm.sum(axis=0),
                     decimals=2)
    rec = np.around(np.diag(cm).astype(float) * 100 / cm.sum(axis=1),
                    decimals=2)

    cm_full = np.vstack((cm, prec))  # adding precision row
    cm_full = np.hstack((cm_full, (
        np.append(rec, np.around(acc * 100, decimals=2))).reshape(len(cm_full),
                                                                  1)))  # adding
    # recall column & total accuracy

    prec_macro = precision_score(y_test, y_pred, average='weighted')
    rec_macro = recall_score(y_test, y_pred, average='weighted')

    print('Accuracy: ', np.around(acc * 100, decimals=2))
    print('Precision: ', round(np.mean(prec), 2))
    print('Recall: ', round(np.mean(rec), 2))
    print('Macro Precision: ', round(prec_macro * 100, 2))
    print('Macro Recall: ', round(rec_macro * 100, 2))
    print(
        'Confusion Matrix (Activities: Walking, Upstairs, Downstairs, '
        'Standing, Sitting, Laying')
    print(cm)
    print(
        'Confusion Matrix & Scores (Actual Activities & Precision vs. '
        'Predicted Activies & Recall; Total Accuracy)')
    print(cm_full)

    return acc, prec_macro, rec_macro, cm, cm_full


def svm_rbf_grid_search():
    data = Data()
    data.load_data(shfl=True)
    cls = Classifiers(data)
    # Default c = 1 , gamma = 1/no_of_features
    param_grid = {'C': [1, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [1 / data.X_train.shape[1], 0.0001, 0.0005, 0.001,
                            0.005, 0.01, 0.1], }

    linear_svc = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'),
                              param_grid)
    cls.add_classifier("rbf-support-vector-machine", linear_svc)
    cls.fit()
    print("Best estimator found by grid search:")
    print(cls.cls['rbf-support-vector-machine'].best_estimator_)
    cls.predict()
    cls.get_scores()
    cls.print_scores()
    pprint.pprint(confusion_matrix(cls.y_test,
                                   cls.y_predict['rbf-support-vector-machine']))
    print(classification_report(cls.y_test,
                                cls.y_predict['rbf-support-vector-machine'],
                                target_names=data.labels))

    get_scores(cls.y_test,
               cls.y_predict['rbf-support-vector-machine'])
    # cls.show_plt()


def plt_all():
    figure = plt.figure(figsize=(27, 9))
    i = 1

    data = Data()
    data.load_data(shfl=True)
    cls = Classifiers(data)

    rfc = RandomForestClassifier(max_features=1)
    cls.add_classifier("random-forest", rfc)

    x_min, x_max = data.X_train.min() - .5, data.X_train.max() + .5
    y_min, y_max = data.y_train.min() - .5, data.y_train.max() + .5
    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # iterate over classifiers
    for classifier, clf in cls.cls.iteritems():
        ax = plt.subplot(1, len(cls.cls) + 1, i)
        clf.fit(data.X_train, data.y_train)
        score = clf.score(data.X_test, data.y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(data.X_train[:, 0], data.X_train[:, 1], c=data.y_train,
                   cmap=cm_bright)
        # and testing points
        ax.scatter(data.X_test[:, 0], data.X_test[:, 1], c=data.y_test,
                   cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        # if ds_cnt == 0:
        #     ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1


def pca():
    pca_values = [2, 5, 10, 25, 30, 40, 50, 100, 150, 200, 250, 300, 561]
    # pca_values = [561]

    scores = {}

    for v in pca_values:
        data = Data()
        # data.load_data(3000)
        data.load_data(shfl=True)
        print("Running PCA with %s components" % v)
        pca = PCA(n_components=v)
        data.X_train = pca.fit_transform(data.X_train)
        data.X_test = pca.transform(data.X_test)
        cls = Classifiers(data)
        dtc = DecisionTreeClassifier()
        gnb = GaussianNB()
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()
        linear_svc = svm.SVC(kernel='linear', class_weight='balanced')
        poly_svc = svm.SVC(kernel='poly', class_weight='balanced')
        rbf_svc = svm.SVC(kernel='rbf', class_weight='balanced')
        knn = KNeighborsClassifier(12)
        rfc = RandomForestClassifier(n_estimators=10, max_features=0.8)
        cls.add_classifier("knn", knn)
        cls.add_classifier("decision-trees", dtc)
        cls.add_classifier("gaussian-naive-bayes", gnb)
        cls.add_classifier("linear-discriminant-analysis", lda)
        cls.add_classifier("quadratic-discriminant-analysis", qda)
        cls.add_classifier("linear-support-vector-machine", linear_svc)
        cls.add_classifier("poly-support-vector-machine", poly_svc)
        cls.add_classifier("rbf-support-vector-machine", rbf_svc)
        cls.add_classifier("random-forest", rfc)

        cls.fit()
        cls.predict()
        cls.get_scores()
        cls.print_scores()
        for k, v in cls.scores.items():
            try:
                scores[k] += [v["accuracy"]]
            except KeyError:
                scores[k] = [v["accuracy"]]

if __name__ == "__main__":
    # cls_compare_shuff()
    # cls_compare_no_shuff()
    # svm_rbf_grid_search()
    svm_linear_grid_search()
    # svm_all_weighted()
    # svm_all_unweighted()
    # cls_compare_pca()
    # plt_all()
    # plt.show()
