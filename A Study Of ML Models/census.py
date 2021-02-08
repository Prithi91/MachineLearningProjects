import csv
import sys
import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import preprocessing
from sklearn.model_selection import learning_curve,train_test_split

def generate_learningcurves(trainx, trainy):
    train_sizes = [5000, 7000, 10000, 15000, 20000, 22000, 24000, 26000]
    trainsizes, train_scores, validation_scores,fit_times,score_times = learning_curve(estimator=tree.DecisionTreeClassifier(max_depth=6),
                                                                 X=trainx, y=trainy, scoring='roc_auc', train_sizes=train_sizes,
                                                                 cv=5, shuffle=True, return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    fit_times_mean = fit_times.mean(axis=1)
    plt.title('DecisionTree LearningCurve-Census')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('ROCAUCScore')
    plt.axis([5000, 26000, 0.8, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('DecisionTreeLearningCurve-Census.png')
    plt.clf()
    trainsizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
        estimator=MLPClassifier(solver='adam', activation="logistic", hidden_layer_sizes=(10,), random_state=1),
                                                                     X=trainx, y=trainy, scoring='roc_auc', train_sizes=train_sizes,
                                                                     cv=5, shuffle=True,return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    fir_scores_mean = fit_times.mean(axis=1)
    plt.title('NeuralNets LearningCurve-Census')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('ROCAUCScore')
    plt.axis([5000, 26000, 0.8, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('NeuralNetsLearningCurve-Census.png')
    plt.clf()
    trainsizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
            estimator=KNeighborsClassifier(n_neighbors=15),
            X=trainx, y=trainy, scoring='roc_auc', train_sizes=train_sizes,
            cv=5, shuffle=True, return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    fit_Scores_mean = fit_times.mean(axis=1)
    plt.title('KNN LearningCurve-Census')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('ROCAUCScore')
    plt.axis([5000, 26000, 0.8, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('KNNLearningCurve-Census.png')
    plt.clf()
    trainsizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
        estimator=AdaBoostClassifier(n_estimators=50, base_estimator=tree.DecisionTreeClassifier(max_depth=3),
                                     learning_rate=0.2),
        X=trainx, y=trainy, scoring='roc_auc', train_sizes=train_sizes,
        cv=5, shuffle=True, return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    fit_times_mean = fit_times.mean(axis=1)
    plt.title('BoostedTree LearningCurve-Census')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('ROCAUCScore')
    plt.axis([5000, 26000, 0.8, 1.0])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('BoostedTreeLearningCurve-Census.png')
    plt.clf()
    trainsizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
        estimator=SVC(kernel='rbf', gamma=0.03),
        X=trainx, y=trainy,scoring='roc_auc', train_sizes=train_sizes,
        cv=5, shuffle=True, return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('SVMrbfKernel LearningCurve-Census')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('ROCAUCScore')
    plt.axis([5000, 26000, 0.8, 1.2])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('SVMRBFLearningCurve-Census.png')
    plt.clf()
    trainx = trainx.astype(np.float64)
    trainsizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
        estimator=LinearSVC(random_state=1, tol=1e-50, class_weight='balanced', max_iter=20000),
        X=trainx, y=trainy, scoring='roc_auc', train_sizes=train_sizes,
        cv=5, shuffle=True, return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.title('SVMLinear LearningCurve-Census')
    plt.xlabel('TrainingSetSize')
    plt.ylabel('ROCAUCScore')
    plt.axis([5000, 26000, 0.8, 1.2])
    plt.plot(train_sizes, train_scores_mean, label="LearningCurve-TrainingSet")
    plt.plot(train_sizes, validation_scores_mean, label="LearningCurve-ValidationSet")
    plt.legend()
    plt.savefig('SVMLinearLearningCurve-Census.png')
    plt.clf()

def generate_performancecurves_allFeatures(trainx, trainy):
    depths = np.linspace(5, 15, 6, endpoint=True)
    res_train = []
    for depth in depths:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf.fit(trainx, trainy)
        ypred_in = clf.predict(trainx)
        f1_train = f1_score(trainy, ypred_in, pos_label=' 0')
        res_train.append(f1_train)
    plt.title('DecisionTree Performance-census')
    plt.xlabel('Tree Depth')
    plt.ylabel('F1Score')
    plt.axis([5, 15, 0.6, 1.0])
    plt.plot(depths, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('DecisionTreePerformance-Census(AF).png')
    plt.clf()
    sizes = [10, 20, 30, 50, 70, 90, 110]
    trainy_float= trainy.astype(np.float64)
    res_train = []
    for size in sizes:
        clf = MLPClassifier(solver='adam', activation="logistic", hidden_layer_sizes=(size,), random_state=1)
        clf.fit(trainx, trainy_float)
        ypred_in = clf.predict(trainx)
        acc_train = f1_score(trainy_float,ypred_in,pos_label=0.)
        res_train.append(acc_train)
    plt.title('NeuralNets Performance-Census')
    plt.xlabel('Number of Neurons')
    plt.ylabel('F1Score')
    plt.axis([10, 110, 0.1, 0.9])
    plt.plot(sizes, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('NeuralNetsPerformance-Census(AF).png')
    plt.clf()
    neighbors = [13, 14, 15, 16, 17, 18, 19, 20]
    res_train = []
    for n in neighbors:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(trainx, trainy_float)
        ypred_in = clf.predict(trainx)
        acc_train = f1_score(trainy_float, ypred_in, pos_label=0.)
        res_train.append(acc_train)
    plt.title('KNN Performance-Census')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('F1Score')
    plt.axis([13, 20, 0.4, 0.9])
    plt.plot(neighbors, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('KNNPerformance-Census(AF).png')
    plt.clf()
    estims = np.linspace(30, 100, 6, endpoint=True)
    res_train = []
    for est in estims:
        clf = AdaBoostClassifier(n_estimators=int(est), base_estimator=tree.DecisionTreeClassifier(max_depth=2),
                                 learning_rate=0.3)
        clf = clf.fit(trainx, trainy)
        pred_in = clf.predict(trainx)
        acc_train = f1_score(trainy, pred_in, pos_label=' 0')
        res_train.append(acc_train)
    plt.title('BoostedTree Performance(LearningRate 0.3)-census')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1-Score')
    plt.axis([30, 100, 0.6, 0.9])
    plt.plot(estims, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('BoostedTreePerformance-Census(AF).png')
    plt.clf()
    gammas = np.linspace(pow(10, -5), 0.1, 5, endpoint=True)
    res_train = []
    for gamma in gammas:
        clf = SVC(kernel='rbf',gamma=gamma,cache_size=7000)
        clf = clf.fit(trainx, trainy)
        pred_in = clf.predict(trainx)
        acc_train = f1_score(trainy, pred_in, pos_label=' 0')
        res_train.append(acc_train)
    plt.title('SVM Performance-Census,kernel:rbf')
    plt.xlabel('Gamma')
    plt.ylabel('F1Score')
    plt.axis([pow(10,-5), 0.1, 0.1, 1.0])
    plt.plot(gammas, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('SVMRBFPerformance-Census(AF).png')
    plt.clf()
    iters = [10000, 20000, 30000, 40000, 50000, 60000]
    res_train = []
    for iter in iters:
        clf = LinearSVC(random_state=1, tol=1e-50, class_weight='balanced', max_iter=iter)
        clf = clf.fit(x_resampled,y_resampled)
        pred_in = clf.predict(x_resampled)
        acc_train = f1_score(y_resampled, pred_in, pos_label=' 0')
        res_train.append(acc_train)
    plt.title('SVM Performance-Census, kernel:linear')
    plt.xlabel('Max Iterations')
    plt.ylabel('F1Score')
    plt.axis([10000, 60000, 0.1, 1.0])
    plt.plot(iters, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('SVMLinearPerformance-Census(AF).png')
    plt.clf()

def get_best_features():
    data = []
    with open('adult.data.csv') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            i = 0
            att = []
            for col in row:
                # print(row[i])
                att.append(row[i])
                i += 1
            data.append(att)
    data_np = np.array(data)
    data_xtrain = data_np[:, 0:-1]
    train_y = data_np[:, -1]
    train_x = data_xtrain[:, 0]
    train_x = np.vstack([train_x, data_xtrain[:, 4]])
    train_x = np.vstack([train_x, data_xtrain[:, 10]])
    train_x = np.vstack([train_x, data_xtrain[:, 11]])
    train_x = np.vstack([train_x, data_xtrain[:, 12]])
    train_x = train_x.transpose()
    string_x = data_np[:, 1]
    string_x = np.vstack([string_x, data_xtrain[:, 5]])
    string_x = np.vstack([string_x, data_xtrain[:, 6]])
    string_x = np.vstack([string_x, data_xtrain[:, 7]])
    string_x = np.vstack([string_x, data_xtrain[:, 13]])
    string_x = string_x.transpose()
    data = []
    with open('adult.test.csv') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            i = 0
            att = []
            for col in row:
                # print(row[i])
                att.append(row[i])
                i += 1
            data.append(att)
    data_np = np.array(data)
    data_xtest = data_np[:, 0:-1]
    test_y = data_np[:, -1]
    test_x = data_xtest[:, 0]
    test_x = np.vstack([test_x, data_xtest[:, 4]])
    test_x = np.vstack([test_x, data_xtest[:, 10]])
    test_x = np.vstack([test_x, data_xtest[:, 11]])
    test_x = np.vstack([test_x, data_xtest[:, 12]])
    test_x = test_x.transpose()
    string_testx = data_np[:, 1]
    string_testx = np.vstack([string_testx, data_xtest[:, 5]])
    string_testx = np.vstack([string_testx, data_xtest[:, 6]])
    string_testx = np.vstack([string_testx, data_xtest[:, 7]])
    string_testx = np.vstack([string_testx, data_xtest[:, 13]])
    string_testx = string_testx.transpose()
    alldata = string_x.copy()
    alldata = np.concatenate((alldata, string_testx))
    enc = preprocessing.OneHotEncoder()
    encoded_x = enc.fit_transform(alldata)
    encoded_trainx = encoded_x.toarray()[0:32561, :]
    encoded_trainx = np.hstack([encoded_trainx, train_x.T[0].reshape(-1, 1)])
    for i in range(1, 5):
        encoded_trainx = np.hstack([encoded_trainx, train_x.T[i].reshape(-1, 1)])
    encoded_testx = encoded_x.toarray()[32561:, :]
    encoded_testx = np.hstack([encoded_testx, test_x.T[0].reshape(-1, 1)])
    for i in range(1, 5):
        encoded_testx = np.hstack([encoded_testx, test_x.T[i].reshape(-1, 1)])
    encoded_trainx = encoded_trainx.astype(np.float64)
    encoded_testx = encoded_testx.astype(np.float64)
    ros = RandomOverSampler(random_state=1)
    x_resampled, y_resampled = ros.fit_resample(encoded_trainx, train_y)
    return x_resampled, y_resampled, encoded_testx, test_y

def generate_performancecurves_bestFeatures(trainx, trainy):
    depths = np.linspace(5, 15, 6, endpoint=True)
    res_train = []
    for depth in depths:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf.fit(trainx, trainy)
        ypred_in = clf.predict(trainx)
        f1_train = f1_score(trainy, ypred_in, pos_label=' 0')
        res_train.append(f1_train)
    plt.title('DecisionTree Performance-census')
    plt.xlabel('Tree Depth')
    plt.ylabel('F1Score')
    plt.axis([5, 15, 0.6, 1.0])
    plt.plot(depths, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('DecisionTreePerformance-Census(BF).png')
    plt.clf()
    sizes = [10, 20, 30, 50, 70, 90, 110]
    res_train = []
    trainy_float = trainy.astype(np.float64)
    for size in sizes:
        clf = MLPClassifier(solver='adam', activation="logistic", hidden_layer_sizes=(size,), random_state=1)
        clf.fit(trainx, trainy_float)
        ypred_in = clf.predict(trainx)
        acc_train = f1_score(trainy_float, ypred_in, pos_label=0.)
        res_train.append(acc_train)
    plt.title('NeuralNets Performance-Census')
    plt.xlabel('Number of Neurons')
    plt.ylabel('F1Score')
    plt.axis([10, 110, 0.1, 0.9])
    plt.plot(sizes, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('NeuralNetsPerformance-Census(BF).png')
    plt.clf()
    neighbors = [13, 14, 15, 16, 17, 18, 19, 20]
    res_train = []
    for n in neighbors:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(trainx, trainy_float)
        ypred_in = clf.predict(trainx)
        acc_train = f1_score(trainy_float, ypred_in, pos_label=0.)
        res_train.append(acc_train)
    plt.title('KNN Performance-Census')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('F1Score')
    plt.axis([13, 20, 0.4, 0.9])
    plt.plot(neighbors, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('KNNPerformance-Census(BF).png')
    plt.clf()
    estims = np.linspace(30, 100, 6, endpoint=True)
    res_train = []
    for est in estims:
        clf = AdaBoostClassifier(n_estimators=int(est), base_estimator=tree.DecisionTreeClassifier(max_depth=2),
                                 learning_rate=0.3)
        clf = clf.fit(trainx, trainy)
        pred_in = clf.predict(trainx)
        acc_train = f1_score(trainy, pred_in, pos_label=' 0')
        res_train.append(acc_train)
    plt.title('BoostedTree Performance(LearningRate 0.3)-census')
    plt.xlabel('Number of Estimators')
    plt.ylabel('F1-Score')
    plt.axis([30, 100, 0.6, 0.9])
    plt.plot(estims, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('BoostedTreePerformance-Census(BF).png')
    plt.clf()
    gammas = np.linspace(pow(10, -5), 0.1, 5, endpoint=True)
    res_train = []
    for gamma in gammas:
        clf = SVC(kernel='rbf', gamma=gamma, cache_size=7000)
        clf = clf.fit(trainx, trainy)
        pred_in = clf.predict(trainx)
        acc_train = f1_score(trainy, pred_in, pos_label=' 0')
        res_train.append(acc_train)
    plt.title('SVM Performance-Census,kernel:rbf')
    plt.xlabel('Gamma')
    plt.ylabel('F1Score')
    plt.axis([pow(10, -5), 0.1, 0.1, 1.0])
    plt.plot(gammas, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('SVMRBFPerformance-Census(BF).png')
    plt.clf()
    iters = [10000, 20000, 30000, 40000, 50000, 60000]
    res_train = []
    for iter in iters:
        clf = LinearSVC(random_state=1, tol=1e-50, class_weight='balanced', max_iter=iter)
        clf = clf.fit(x_resampled, y_resampled)
        pred_in = clf.predict(x_resampled)
        acc_train = f1_score(y_resampled, pred_in, pos_label=' 0')
        res_train.append(acc_train)
    plt.title('SVM Performance-Census, kernel:linear')
    plt.xlabel('Max Iterations')
    plt.ylabel('F1Score')
    plt.axis([10000, 60000, 0.1, 1.0])
    plt.plot(iters, res_train, label="F1Score-Train")
    plt.legend()
    plt.savefig('SVMLinearPerformance-Census(BF).png')
    plt.clf()

def generate_ROCcurves(trainx,trainy,testx,testy):
    clf = tree.DecisionTreeClassifier(max_depth=6)
    prob = clf.fit(trainx, trainy).predict_proba(testx)
    fpr, tpr, thresholds = roc_curve(testy, prob[:,0],pos_label=' 0')
    roc_auc = auc(fpr, tpr)
    plt.title("DecisionTreeROC AUC- Census")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0.0,1.0,0.0,1.05])
    plt.plot(fpr, tpr, label='ROC curve(area = %0.2f)' %roc_auc)
    plt.plot([0,1],[0,1], color='black',linestyle='--')
    plt.legend()
    plt.savefig('DecisionTreeROC-Census.png')
    plt.clf()
    trainy_float = trainy.astype(np.float64)
    testy_float = testy.astype(np.float64)
    clf = MLPClassifier(solver='adam', activation="logistic", hidden_layer_sizes=(10,), random_state=1)
    prob = clf.fit(trainx, trainy_float).predict_proba(testx)
    fpr, tpr, thresholds = roc_curve(testy_float, prob[:, 0], pos_label=0.)
    roc_auc = auc(fpr, tpr)
    plt.title("NeuralNetsROC AUC- Census")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.plot(fpr, tpr, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.savefig('NeuralNetsROC-Census.png')
    plt.clf()
    clf = KNeighborsClassifier(n_neighbors=15)
    prob = clf.fit(trainx, trainy_float).predict_proba(testx)
    fpr, tpr, thresholds = roc_curve(testy_float, prob[:, 0], pos_label=0.)
    roc_auc = auc(fpr, tpr)
    plt.title("KNNROC AUC- Census")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.plot(fpr, tpr, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.savefig('KNNROC-Census.png')
    plt.clf()
    clf = AdaBoostClassifier(n_estimators=50, base_estimator=tree.DecisionTreeClassifier(max_depth=2),
                             learning_rate=0.3)
    prob = clf.fit(trainx, trainy).predict_proba(testx)
    fpr, tpr, thresholds = roc_curve(testy, prob[:, 0], pos_label=' 0')
    roc_auc = auc(fpr, tpr)
    plt.title("BoostedTreeROC AUC- Census")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.plot(fpr, tpr, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.savefig('BoostedTreeROC-Census.png')
    plt.clf()
    clf = SVC(kernel='rbf', gamma=0.03, cache_size=7000, probability=True)
    prob = clf.fit(trainx, trainy).predict_proba(testx)
    fpr, tpr, thresholds = roc_curve(testy, prob[:, 0], pos_label=' 0')
    roc_auc = auc(fpr, tpr)
    plt.title("SVMROC AUC- Census, kernel:rbf")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.plot(fpr, tpr, label='ROC  curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.savefig('SVMRBFROC-Census.png')
    plt.clf()
    svm = LinearSVC(random_state=1, tol=1e-50, class_weight='balanced', max_iter=20000)
    clf =CalibratedClassifierCV(svm)
    prob = clf.fit(trainx,trainy).predict_proba(testx)
    fpr, tpr, thresholds = roc_curve(testy, prob[:,0], pos_label=' 0')
    roc_auc = auc(fpr, tpr)
    plt.title("SVMROC AUC- Census, kernel:linear")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.plot(fpr, tpr, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.savefig('SVMLinearROC-Census.png')
    plt.clf()

if __name__=="__main__":
    data = []
    with open('adult.data.csv') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            i = 0
            att = []
            for col in row:
                # print(row[i])
                att.append(row[i])
                i += 1
            data.append(att)
    data_np = np.array(data)
    data_xtrain = data_np[:, 0:-1]
    train_y = data_np[:, -1]
    ##Data Pre-Processing
    train_x = data_xtrain[:, 0]
    train_x = np.vstack([train_x, data_xtrain[:, 2]])
    train_x = np.vstack([train_x, data_xtrain[:, 4]])
    train_x = np.vstack([train_x, data_xtrain[:, 10]])
    train_x = np.vstack([train_x, data_xtrain[:, 11]])
    train_x = np.vstack([train_x, data_xtrain[:, 12]])
    train_x = train_x.transpose()
    string_x = data_np[:, 1]
    string_x = np.vstack([string_x, data_xtrain[:, 3]])
    string_x = np.vstack([string_x, data_xtrain[:, 5]])
    string_x = np.vstack([string_x, data_xtrain[:, 6]])
    string_x = np.vstack([string_x, data_xtrain[:, 7]])
    string_x = np.vstack([string_x, data_xtrain[:, 8]])
    string_x = np.vstack([string_x, data_xtrain[:, 9]])
    string_x = np.vstack([string_x, data_xtrain[:, 13]])
    string_x = string_x.transpose()
    data = []
    with open('adult.test.csv') as datafile:
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
    test_y = data_np[:, -1]
    ##Data Pre-Processing
    test_x = data_xtest[:, 0]
    test_x = np.vstack([test_x, data_xtest[:, 2]])
    test_x = np.vstack([test_x, data_xtest[:, 4]])
    test_x = np.vstack([test_x, data_xtest[:, 10]])
    test_x = np.vstack([test_x, data_xtest[:, 11]])
    test_x = np.vstack([test_x, data_xtest[:, 12]])
    test_x = test_x.transpose()
    string_testx = data_np[:, 1]
    string_testx = np.vstack([string_testx, data_xtest[:, 3]])
    string_testx = np.vstack([string_testx, data_xtest[:, 5]])
    string_testx = np.vstack([string_testx, data_xtest[:, 6]])
    string_testx = np.vstack([string_testx, data_xtest[:, 7]])
    string_testx = np.vstack([string_testx, data_xtest[:, 8]])
    string_testx = np.vstack([string_testx, data_xtest[:, 9]])
    string_testx = np.vstack([string_testx, data_xtest[:, 13]])
    string_testx = string_testx.transpose()
    alldata = string_x.copy()
    alldata = np.concatenate((alldata, string_testx))
    enc = preprocessing.OneHotEncoder()
    encoded_x = enc.fit_transform(alldata)
    encoded_trainx = encoded_x.toarray()[0:32561, :]
    encoded_trainx = np.hstack([encoded_trainx, train_x.T[0].reshape(-1, 1)])
    for i in range(1, 6):
        encoded_trainx = np.hstack([encoded_trainx, train_x.T[i].reshape(-1, 1)])
    encoded_testx = encoded_x.toarray()[32561:, :]
    encoded_testx = np.hstack([encoded_testx, test_x.T[0].reshape(-1, 1)])
    for i in range(1, 6):
        encoded_testx = np.hstack([encoded_testx, test_x.T[i].reshape(-1, 1)])
    encoded_trainx = encoded_trainx.astype(np.float64)
    encoded_testx = encoded_testx.astype(np.float64)
    ros = RandomOverSampler(random_state=1)
    x_resampled, y_resampled = ros.fit_resample(encoded_trainx, train_y)
    generate_performancecurves_allFeatures(x_resampled,y_resampled)
    trainx, trainy, testx, testy = get_best_features()
    generate_performancecurves_bestFeatures(trainx, trainy)
    generate_learningcurves(x_resampled, y_resampled)
    generate_ROCcurves(trainx,trainy,testx,testy)


