import csv
import sys
import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, plot_confusion_matrix
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import learning_curve,train_test_split

def generate_learningcurves(x_sampled, y_sampled):
    train_sizes = [2000, 3000, 4000, 5000, 6000, 7000, 8344]
    trainsizes, train_scores, validation_scores = learning_curve(estimator=tree.DecisionTreeClassifier(max_depth=15),
                                                                 X=x_sampled, y=y_sampled, scoring='f1_weighted',
                                                                 train_sizes=train_sizes,
                                                                 cv=5, shuffle=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('DecisionTree LearningCurve-Avila, max_depth:15')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('F1WeightedScore')
    plt.axis([2000, 8344, 0.8, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('DecisionTreeLearningCurve-Avila.png')
    plt.clf()
    trainsizes, train_scores, validation_scores = learning_curve(
        estimator=MLPClassifier(solver='sgd', activation="tanh", hidden_layer_sizes=(80, 30,), random_state=1),
        X=x_sampled, y=y_sampled, scoring='f1_weighted', train_sizes=train_sizes,
        cv=5, shuffle=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('NeuralNets LearningCurve-Avila, number of nodes:(80,30)')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('F1WeightedScore')
    plt.axis([2000, 8344, 0.5, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('NeuralNetsLearningCurve-Avila.png')
    plt.clf()
    trainsizes, train_scores, validation_scores = learning_curve(
        estimator=KNeighborsClassifier(n_neighbors=1),
        X=x_sampled, y=y_sampled, scoring='f1_weighted', train_sizes=train_sizes,
        cv=5, shuffle=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('KNN LearningCurve-Avila, n_neighbors:1')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('F1WeightedScore')
    plt.axis([2000, 8344, 0.7, 1.2])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('KNNLearningCurve-Avila.png')
    plt.clf()
    trainsizes, train_scores, validation_scores = learning_curve(
            estimator=AdaBoostClassifier(n_estimators=45, base_estimator=tree.DecisionTreeClassifier(max_depth=7),
                                         learning_rate=0.2),
            X=x_sampled, y=y_sampled, train_sizes=train_sizes,scoring='f1_weighted',
            cv=5, shuffle=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('BoostedTree LearningCurve-Avila, max_depth:7, n_estimators:45')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('F1WeightedScore')
    plt.axis([2000, 8344, 0.8, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('BoostedTreeLearningCurve-Avila.png')
    plt.clf()
    trainsizes, train_scores, validation_scores = learning_curve(
        estimator=SVC(kernel='rbf', gamma=0.6),
        X=x_sampled, y=y_sampled, scoring='f1_weighted', train_sizes=train_sizes,
        cv=5, shuffle=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('SVMrbfKernel LearningCurve-Avila, gamma:0.6')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('F1WeightedScore')
    plt.axis([2000, 8344, 0.6, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('SVMrbfkernelLearningCurve-Avila.png')
    plt.clf()
    trainsizes, train_scores, validation_scores = learning_curve(
        estimator=SVC(kernel='poly', gamma=0.7),
        X=x_sampled, y=y_sampled, scoring='f1_weighted', train_sizes=train_sizes,
        cv=5, shuffle=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('SVMpolyKernel LearningCurve-Avila, gamma:0.7')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('F1WeightedScore')
    plt.axis([2000, 8344, 0.6, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('SVMpolykernelLearningCurve-Avila.png')
    plt.clf()

def generate_performancecurves(x_sampled, y_sampled):
    depths = [8,9,10,11,12,13,14,15]
    precision_score = []
    for depth in depths:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf.fit(x_sampled, y_sampled)
        pred_in = clf.predict(x_sampled)
        score = f1_score(y_sampled, pred_in, average="weighted")
        precision_score.append(score)
    plt.title('DecisionTree Performance-Avila')
    plt.xlabel('Tree Depth')
    plt.ylabel('F1Score')
    plt.axis([8, 15, 0.8, 1.0])
    plt.plot(depths, precision_score, label="F1Score-Train")
    plt.legend()
    plt.savefig('DecisionTreePerformance-Avila.png')
    plt.clf()
    sizes=[20,30,40,50,60,70,80,90,100]
    precision_score = []
    for size in sizes:
        clf = MLPClassifier(solver='sgd', activation="tanh", hidden_layer_sizes=(size,30), random_state=1)
        clf.fit(x_sampled, y_sampled)
        pred_in = clf.predict(x_sampled)
        score = f1_score(y_sampled, pred_in, average="weighted")
        precision_score.append(score)
    plt.title('NeuralNets Performance-Avila, Second hiddenlayer :30 nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('F1Score')
    plt.axis([20, 100, 0.6, 1.0])
    plt.plot(sizes, precision_score, label="F1Score-Train")
    plt.legend()
    plt.savefig('NeuralNetsPerformance-Avila.png')
    plt.clf()
    neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    precision_score = []
    for n in neighbors:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(x_sampled, y_sampled)
        pred_in = clf.predict(x_sampled)
        score = f1_score(y_sampled, pred_in, average="weighted")
        precision_score.append(score)
    plt.title('KNN Performance-Avila')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('F1Score')
    plt.axis([1, 9, 0.8, 1.0])
    plt.plot(neighbors, precision_score, label="F1Score-Train")
    plt.legend()
    plt.savefig('KNNPerformance-Avila.png')
    plt.clf()
    estims = [30, 40, 50, 60, 70, 80]
    precision_score = []
    for est in estims:
        clf = AdaBoostClassifier(n_estimators=est, base_estimator=tree.DecisionTreeClassifier(max_depth=7),
                                 learning_rate=0.2)
        clf.fit(x_sampled, y_sampled)
        pred_in = clf.predict(x_sampled)
        score = f1_score(y_sampled, pred_in, average="weighted")
        precision_score.append(score)
    plt.title('BoostedTree Performance-Avila(Tree Depth-7)')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1Score')
    plt.axis([30, 80, 0.9, 1.0])
    plt.plot(estims, precision_score, label="F1Score-Train")
    plt.legend()
    plt.savefig('BoostedTreePerformance-Avila.png')
    plt.clf()
    gammas = np.linspace(0.15, 0.9, 8, endpoint=True)
    precision_score=[]
    for gamma in gammas:
        clf = SVC(kernel='rbf', gamma=gamma, probability=True)
        clf.fit(x_sampled, y_sampled)
        pred_in = clf.predict(x_sampled)
        score = f1_score(y_sampled, pred_in, average="weighted")
        precision_score.append(score)
    plt.title('SVM Performance-Avila(Kernel:RBF)')
    plt.xlabel('gamma')
    plt.ylabel('F1Score')
    plt.axis([0.15, 0.9, 0.6, 1.0])
    plt.plot(gammas, precision_score, label="F1Score-Train")
    plt.legend()
    plt.savefig('SVMRBFPerformance-Avila.png')
    plt.clf()
    gammas_2 = np.linspace(0.2,1.0,6, endpoint=True)
    score_svm2=[]
    for gamma in gammas_2:
        clf = SVC(kernel='poly', gamma=gamma, probability=True)
        clf.fit(x_sampled, y_sampled)
        pred_in = clf.predict(x_sampled)
        score = f1_score(y_sampled, pred_in, average="weighted")
        score_svm2.append(score)
    plt.title('SVM Performance-Avila(Kernel:poly)')
    plt.xlabel('gamma')
    plt.ylabel('F1Score')
    plt.axis([0.2, 1.0, 0.8, 1.0])
    plt.plot(gammas_2, score_svm2, label="F1Score-Train")
    plt.legend()
    plt.savefig('SVMPolyPerformance-Avila.png')
    plt.clf()

def generate_confusion_matrices(x_sampled, y_sampled,data_xtest,data_ytest):
    clf = tree.DecisionTreeClassifier(max_depth=15)
    clf.fit(x_sampled, y_sampled)
    disp = plot_confusion_matrix(clf, data_xtest, data_ytest, normalize='true')
    disp.ax_.set_title("Decision Tree ConfusionMatrix")
    plt.savefig("Decision Tree ConfusionMatrix.png")
    plt.clf()
    clf = MLPClassifier(solver='sgd', activation="tanh", hidden_layer_sizes=(80, 30), random_state=1)
    clf.fit(x_sampled, y_sampled)
    disp = plot_confusion_matrix(clf, data_xtest, data_ytest, normalize='true')
    disp.ax_.set_title("NeuralNets ConfusionMatrix")
    plt.savefig("NeuralNets ConfusionMatrix.png")
    plt.clf()
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_sampled, y_sampled)
    disp = plot_confusion_matrix(clf, data_xtest, data_ytest, normalize='true')
    disp.ax_.set_title("KNN ConfusionMatrix")
    plt.savefig("Knn ConfusionMatrix.png")
    plt.clf()
    clf = AdaBoostClassifier(n_estimators=45, base_estimator=tree.DecisionTreeClassifier(max_depth=7),
                             learning_rate=0.2)
    clf.fit(x_sampled, y_sampled)
    disp = plot_confusion_matrix(clf, data_xtest, data_ytest, normalize='true')
    disp.ax_.set_title("BoostedTree ConfusionMatrix")
    plt.savefig("BoostedTree ConfusionMatrix.png")
    plt.clf()
    clf = SVC(kernel='rbf', gamma=0.6, probability=True)
    clf.fit(x_sampled, y_sampled)
    disp = plot_confusion_matrix(clf, data_xtest, data_ytest, normalize='true')
    disp.ax_.set_title("SVMRBFkernel ConfusionMatrix")
    plt.savefig("SVMRBFKernel ConfusionMatrix.png")
    plt.clf()
    clf = SVC(kernel='poly', gamma=0.7, probability=True)
    clf.fit(x_sampled, y_sampled)
    disp = plot_confusion_matrix(clf, data_xtest, data_ytest, normalize='true')
    disp.ax_.set_title("SVMPolykernel ConfusionMatrix")
    plt.savefig("SVMPolyKernel ConfusionMatrix.png")
    plt.clf()

if __name__=="__main__":
    data = []
    with open('avila/avila-tr.csv') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            i = 0
            att = []
            for col in row:
                att.append(row[i])
                i += 1
            data.append(att)
    data_np = np.array(data)
    data_xtrain = data_np[:, 0:-1]
    train_y = data_np[:, -1]
    clf = tree.DecisionTreeClassifier(max_depth=15)
    clf = clf.fit(data_xtrain, train_y)
    data = []
    with open('avila/avila-ts.csv') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            i = 0
            att = []
            for col in row:
                att.append(row[i])
                i += 1
            data.append(att)
    data_np = np.array(data)
    data_xtest = data_np[:, 0:-1]
    data_ytest = data_np[:, -1]
    data_xtrain = data_xtrain.astype(np.float64)
    data_xtest = data_xtest.astype(np.float64)
    sm = SMOTE(random_state=1, k_neighbors=4)
    x_sampled, y_sampled = sm.fit_sample(data_xtrain, train_y)
    generate_performancecurves(x_sampled, y_sampled)
    generate_learningcurves(data_xtrain, train_y)
    generate_confusion_matrices(x_sampled,y_sampled,data_xtest,data_ytest)
