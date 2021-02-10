import csv
import numpy as np
import pandas as pd
#pd.set_option('display.max_rows',None)
import matplotlib
matplotlib.use('Agg')
import re
import math
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score, silhouette_score
from sklearn.metrics import mean_squared_error, f1_score,plot_confusion_matrix
import seaborn as sb
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import learning_curve, train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import preprocessing
from scipy.stats import kurtosis

def get_data():
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
    return x_sampled, y_sampled

def clustering_kmeans(trainx, trainy, isreduced=False, alg='pca', finalclusters=2, xlim=[], ylim=[]):
    n_clusters = range(2, 15)
    scores = []
    homogeneity_scores = []
    for i in n_clusters:
        kclusters = KMeans(n_clusters=i, n_jobs=16)
        kclusters.fit(trainx)
        scores.append(silhouette_score(trainx, kclusters.labels_))
        #if(isreduced==False):
        homogeneity_scores.append(homogeneity_score(trainy, kclusters.labels_))
    plt.plot(n_clusters, scores)
    plt.xlabel('K values')
    plt.ylabel('Average Silhouette Scores')
    plt.title('Silhouette Method')
    if(isreduced==False):
        plt.savefig('Silhouette Method-Avila.png')
    else:
        plt.savefig("Silhouette Method-Avila(DimRed{alg}_{nc}).png".format(alg=alg, nc=finalclusters))
    plt.clf()
    if(isreduced==False):
        plt.plot(n_clusters, homogeneity_scores)
        plt.xlabel('K values')
        plt.ylabel('Scores')
        plt.title('Homogeneity Scores')
        plt.savefig('Homogeneity Scores-Avila.png')
    else:
        plt.plot(n_clusters, homogeneity_scores)
        plt.xlabel('K values')
        plt.ylabel('Scores')
        plt.title('Homogeneity Scores')
        plt.savefig("Homogeneity Scores-Avila(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters))
    plt.clf()
    kclusters_final = KMeans(n_clusters=finalclusters, n_jobs=8)
    kclusters_final = kclusters_final.fit(trainx)
    # # print(kclusters_final.labels_)
    # labels = kclusters_final.labels_
    trainx['kmeans_cluster'] = kclusters_final.labels_
    trainx.sort_values('kmeans_cluster', axis=0, inplace=True)
    values = [0, 1, 2]
    # colors = cm.rainbow(np.linspace(0,1,4))
    colors = ['blue', 'red', 'green']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if(isreduced==False):
        ax.set_xlim([-5, 50])
        ax.set_ylim([-5, 10])
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_title('Scatter Plot - kmeans clustered data')
    for v, color in zip(values, colors):
        indices = trainx['kmeans_cluster'] == v
        if(isreduced==False):
            ax.scatter(trainx.loc[indices, 1], trainx.loc[indices, 2], c=color)
        else:
            if(alg=='pca'): ax.scatter(trainx.loc[indices, 0], trainx.loc[indices, 1], c=color)
            if(alg =='ica'): ax.scatter(trainx.loc[indices, 0], trainx.loc[indices, 4], c=color)
            if(alg == 'rp'): ax.scatter(trainx.loc[indices,4], trainx.loc[indices,6], c=color)
            if(alg== 'svd'): ax.scatter(trainx.loc[indices,8], trainx.loc[indices,7], c=color)
    if(isreduced==False):
        plt.savefig("Scatter Plot-Clustered_{nc}".format(alg=alg, nc=finalclusters))
    else:
        plt.savefig("Scatter Plot- {alg}(Clustered)_{nc}".format(alg=alg,nc=finalclusters))
    plt.clf()
    # kclusters_final = KMeans(n_clusters=finalclusters, n_jobs=8)
    # kclusters_final = kclusters_final.fit(trainx)
    #trainx['kmeans_cluster'] = kclusters_final.labels_
    labels = pd.DataFrame(kclusters_final.labels_, columns=['Clusters'])
    labels =labels.replace(1,10)
    labels.sort_values('Clusters', axis=0, inplace=True)
    labels['truelabels'] = list(trainy)
    labels = labels.replace("A",0)
    labels = labels.replace("B",1)
    labels = labels.replace("C",2)
    labels = labels.replace("D",3)
    labels = labels.replace("E",4)
    labels = labels.replace("F",5)
    labels = labels.replace("G",6)
    labels = labels.replace("H",7)
    labels = labels.replace("I",8)
    labels = labels.replace("W",9)
    labels = labels.replace("X",10)
    labels = labels.replace("Y",11)
    labels.index.name = 'Clusters'
    ax= plt.axes()
    sb.heatmap(labels, ax=ax)
    ax.set_title("Kmeans- Clustering")
        #fig = map.get_figure()
    if (isreduced == False):
        plt.savefig('Heatmap_Avila(clusteringKM).png', dpi=400)
        plt.clf()
    else:
        plt.savefig("Heatmap_Avila(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters), dpi=400)
        plt.clf()
    return kclusters_final.labels_

def clustering_em(trainx, trainy, isreduced=False, alg='pca', finalclusters=2,xlim=[], ylim=[]):
    #processed_numerical_p50, merged = get_data()
    n_clusters = range(2, 15)
    scores = []
    homogeneity_scores = []
    for i in n_clusters:
        gmm = GaussianMixture(n_components=i, covariance_type='full')
        kclusters = gmm.fit(trainx)
        y_pred = gmm.predict(trainx)
        scores.append(silhouette_score(trainx, y_pred))
        homogeneity_scores.append(homogeneity_score(trainy, y_pred))
    plt.plot(n_clusters, scores)
    plt.xlabel('K values')
    plt.ylabel('Average Silhouette Scores')
    plt.title('Silhouette Method')
    if(isreduced==False):
        plt.savefig('Silhouette Method(EM)-Avila.png')
    else:
        plt.savefig("Silhouette Method(EM)-Avila(DimRed{alg}_{nc}).png".format(alg=alg, nc=finalclusters))
    plt.clf()
    plt.plot(n_clusters, homogeneity_scores)
    plt.xlabel('K values')
    plt.ylabel('Scores')
    plt.title('Homogeneity Scores')
    if(isreduced==False):
        plt.savefig('Homogeneity Scores(EM)-Avila.png')
    else:
        plt.savefig("Homogeneity Scores(EM)-Avila(DimRed{alg}_{nc}).png".format(alg=alg, nc=finalclusters))
    plt.clf()
    gmm_final = GaussianMixture(n_components=finalclusters, covariance_type='full')
    kclusters_final = gmm_final.fit(trainx)
    y_pred = gmm_final.predict(trainx)
    labels = y_pred
    # print(kclusters_final.labels_)
    trainx['kmeans_cluster'] = y_pred
    trainx.sort_values('kmeans_cluster', axis=0, inplace=True)
    values = [0, 1, 2]
    # colors = cm.rainbow(np.linspace(0,1,4))
    colors = ['blue', 'red', 'green']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if (isreduced == False):
        ax.set_xlim([-5, 50])
        ax.set_ylim([-5, 10])
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_title('Scatter Plot - EM clustered data')
    for v, color in zip(values, colors):
        indices = trainx['kmeans_cluster'] == v
        if (isreduced == False):
            ax.scatter(trainx.loc[indices, 1], trainx.loc[indices, 2], c=color)
        else:
            if (alg == 'pca'): ax.scatter(trainx.loc[indices, 0], trainx.loc[indices, 1], c=color)
            if (alg == 'ica'): ax.scatter(trainx.loc[indices, 0], trainx.loc[indices, 4], c=color)
    if (isreduced == False):
        plt.savefig("Scatter Plot(EM)-Clustered_{nc}".format(alg=alg, nc=finalclusters))
    else:
        plt.savefig("Scatter Plot(EM)- {alg}(Clustered)_{nc}".format(alg=alg, nc=finalclusters))
    plt.clf()
    #trainx['kmeans_cluster'] = y_pred
    labels = pd.DataFrame(y_pred, columns=['Clusters'])
    labels =labels.replace(1,10)
    labels = labels.replace(2,50)
    labels.sort_values('Clusters', axis=0, inplace=True)
    labels['truelabels'] = list(trainy)
    labels = labels.replace("A",0)
    labels = labels.replace("B",1)
    labels = labels.replace("C",2)
    labels = labels.replace("D",3)
    labels = labels.replace("E",4)
    labels = labels.replace("F",5)
    labels = labels.replace("G",6)
    labels = labels.replace("H",7)
    labels = labels.replace("I",8)
    labels = labels.replace("W",9)
    labels = labels.replace("X",10)
    labels = labels.replace("Y",11)
    labels.index.name = 'Clusters'
    ax= plt.axes()
    sb.heatmap(labels, ax=ax)
    ax.set_title("EM- Clustering")
        #fig = map.get_figure()
    if (isreduced == False):
        plt.savefig('Heatmap(EM)_Avila(clustering).png', dpi=400)
        plt.clf()
    else:
        plt.savefig("Heatmap(EM)_Avila(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters), dpi=400)
    return labels

def dim_red_pca(trainx, trainy):
    trainx = StandardScaler().fit_transform(trainx)
    pca = PCA(n_components=7)
    components = pca.fit_transform(trainx)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    #plt.bar(features, pca.explained_variance_ratio_)
    plt.title("Distribution of variance among the components")
    plt.xlabel('PCA Components')
    plt.ylabel('Variance %')
    plt.xticks(features)
    plt.xticks(features)
    plt.savefig("pca_variance.png")
    plt.clf()
    pc = pd.DataFrame(components)
    pc_final = pc.iloc[:,:2]
    df_scatter = pd.DataFrame(data=pc_final)
    df_scatter.rename(columns={0:"Feature1",1:"Feature2"}, inplace=True)
    target_df = pd.DataFrame(data=trainy)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y']
    colors = cm.rainbow(np.linspace(0, 1, len(values)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_xlim([-2,6])
    ax.set_ylim([-5,8])
    ax.set_title('Scatter Plot - Clustering of labelled Data')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        #ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
        ax.plot(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color, marker='.', ms=6.5,
                linestyle='', )
    plt.savefig("Scatter Plot- pca(Original)")
    plt.clf()
    comp = range(1, 10)
    rmsevals = []
    for i in comp:
        pca = PCA(n_components=i)
        reduced = pca.fit_transform(trainx)
        #inv = np.linalg.pinv(svd.components_)
        reconstructed = pca.inverse_transform(reduced)
        rmse = math.sqrt(mean_squared_error(trainx, reconstructed))
        rmsevals.append(rmse)
    plt.plot(comp, rmsevals)
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE(Reconstruction error)')
    plt.title('Reconstruction error as a function of number of components')
    plt.savefig('RMSE_pca.png')
    plt.clf()
    return pc

def dim_red_ica(trainx):
    n = range(1,10)
    mean_kurt_scores=[]
    for i in n:
        ica = FastICA(n_components=i)
        reduced = ica.fit_transform(trainx)
        # kurt=[]
        # for j in range(i):
        #     kurt.append(kurtosis(reduced[j]))
        #mean_kurt_scores.append(abs(kurtosis(reduced[i])))
        mean_kurt_scores.append(np.mean(kurtosis(reduced)))
    plt.plot(n, mean_kurt_scores)
    plt.title("Kurtosis as a function of number of components")
    plt.xlabel("Number of components")
    plt.ylabel("Mean Kurtosis")
    plt.savefig("ica_kurtosis.png")
    plt.clf()
    rmsevals=[]
    for i in n:
        ica = FastICA(n_components=i)
        reduced =ica.fit_transform(trainx)
        reconstructed = ica.inverse_transform(reduced)
        rmse = math.sqrt(mean_squared_error(trainx, reconstructed))
        rmsevals.append(rmse)
    plt.plot(n,rmsevals)
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE(Reconstruction error)')
    plt.title('Reconstruction error as a function of number of components')
    plt.savefig('RMSE_ica(Avila).png')
    plt.clf()
    ica = FastICA(n_components=8)
    reduced = ica.fit_transform(trainx)
    ic = pd.DataFrame(reduced)
   # fig = sb.pairplot(ic)
    #fig.savefig("Pairplot-Avila(ica-original).png")
    ic_final = pd.concat([ic.iloc[:, 1], ic.iloc[:,5]],axis=1)
    df_scatter = pd.DataFrame(data=ic_final)
    df_scatter.rename(columns={1: "Feature1", 5: "Feature2"}, inplace=True)
    target_df = pd.DataFrame(data=trainy)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y']
    colors = cm.rainbow(np.linspace(0, 1, len(values)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Individual Component 1')
    ax.set_ylabel('Individual Component 2')
    ax.set_xlim([-0.02, 0.04])
    ax.set_ylim([-0.05, 0.15])
    ax.set_title('Scatter Plot - Clustering of labelled Data')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        # ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
        ax.plot(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color, marker='.', ms=6.5,
                linestyle='', )
    plt.savefig("Scatter Plot- ica(Original)")
    plt.clf()
    return ic

def dim_red_rp(trainx):
    comp = range(2,10)
    rmsevals=[]
    for i in comp:
        rp = GaussianRandomProjection(n_components=i)
        reduced = rp.fit_transform(trainx)
        inv = np.linalg.pinv(rp.components_)
        reconstructed = np.dot(reduced, inv.T)
        rmse = math.sqrt(mean_squared_error(trainx, reconstructed))
        rmsevals.append(rmse)
    plt.plot(comp,rmsevals)
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE(Reconstruction error)')
    plt.title('Construction error as a function of number of components')
    plt.savefig('RMSE_rp(Avila).png')
    plt.clf()
    rp = GaussianRandomProjection(n_components=9)
    reduced = rp.fit_transform(trainx)
    rp_x = pd.DataFrame(reduced)
    # fig = sb.pairplot(rp_x)
    # fig.savefig("Pairplot-Avila(rp-original).png")
    rp_final = pd.concat([rp_x.iloc[:, 4], rp_x.iloc[:, 6]], axis=1)
    df_scatter = pd.DataFrame(data=rp_final)
    df_scatter.rename(columns={4: "Feature1", 6: "Feature2"}, inplace=True)
    target_df = pd.DataFrame(data=trainy)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y']
    colors = cm.rainbow(np.linspace(0, 1, len(values)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Individual Component 1')
    ax.set_ylabel('Individual Component 2')
    ax.set_xlim([-10, 30])
    ax.set_ylim([-10, 30])
    ax.set_title('Scatter Plot - Clustering of labelled Data')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        # ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
        ax.plot(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color, marker='.', ms=6.5,
                linestyle='', )
    plt.savefig("Scatter Plot- rp(Original)")
    plt.clf()
    return rp_x

def dim_red_svd(trainx):
    comp = range(2,10)
    rmsevals=[]
    for i in comp:
        svd = TruncatedSVD(n_components=i)
        reduced = svd.fit_transform(trainx)
        inv = np.linalg.pinv(svd.components_)
        reconstructed = np.dot(reduced, inv.T)
        rmse = math.sqrt(mean_squared_error(trainx, reconstructed))
        rmsevals.append(rmse)
    plt.plot(comp, rmsevals)
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE(Reconstruction error)')
    plt.title('Construction error as a function of number of components')
    plt.savefig('RMSE_svd(Avila).png')
    plt.clf()
    svd = TruncatedSVD(n_components=9)
    reduced = svd.fit_transform(trainx)
    svd_x = pd.DataFrame(reduced)
    # fig = sb.pairplot(svd_x)
    # fig.savefig("Pairplot-Avila(svd-original).png")
    svd_final = pd.concat([svd_x.iloc[:, 8], svd_x.iloc[:, 7]], axis=1)
    df_scatter = pd.DataFrame(data=svd_final)
    df_scatter.rename(columns={8: "Feature1", 7: "Feature2"}, inplace=True)
    target_df = pd.DataFrame(data=trainy)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y']
    colors = cm.rainbow(np.linspace(0, 1, len(values)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Individual Component 1')
    ax.set_ylabel('Individual Component 2')
    ax.set_xlim([-6, 10])
    ax.set_ylim([-6, 3])
    ax.set_title('Scatter Plot - Clustering of labelled Data')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        # ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
        ax.plot(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color, marker='.', ms=6.5,
                linestyle='', )
    plt.savefig("Scatter Plot- svd(Original)")
    plt.clf()
    return svd_x

def plot_original_clusters(trainx, trainy):
    data_1 = trainx.iloc[:, 1]
    data_2 = trainx.iloc[:, 2]
    data_final = pd.concat([data_1,data_2], axis=1)
    data_scatter = pd.DataFrame(data=data_final)
    #print(data_scatter)
    data_scatter.rename(columns={1: "Feature1", 2: "Feature2"}, inplace=True)
    #print(data_scatter)
    target_df = pd.DataFrame(data=trainy)
    df_toplot = pd.concat([data_scatter, target_df], axis=1)
    values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'W', 'X', 'Y']
    colors = cm.rainbow(np.linspace(0, 1, len(values)))
    #colors = ['blue', 'red']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim([-5,50])
    ax.set_ylim([-5,10])
    ax.set_title('Pairwise Scatter Plot - Clustering of labelled Data')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
       # ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
        ax.plot(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color, marker='.', ms=10.5,linestyle='', )
    plt.savefig("Scatter Plot-(Original)")
    plt.clf()

def train_NN(reducedx, trainy, alg, testx=None, testy=None):
    sizes = [50, 60, 70, 80, 90, 100,120,140]
    precision_score = []
    test_scores=[]
    for size in sizes:
        clf = MLPClassifier(solver='adam', activation="tanh", hidden_layer_sizes=(size, 30), random_state=1)
        clf.fit(reducedx, trainy)
        pred_in = clf.predict(reducedx)
        score = f1_score(trainy, pred_in, average="weighted")
        precision_score.append(score)
    plt.title('NeuralNets Performance-Avila, Second hiddenlayer :30 nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('F1Score')
    plt.axis([50, 140, 0.6, 1.0])
    plt.plot(sizes, precision_score, label="F1Score-Train")
    #plt.plot(sizes, test_scores, label="F1Score-Test")
    plt.legend()
    plt.savefig("NeuralNetsPerformance-Avila({alg}).png".format(alg=alg))
    plt.clf()
    clf = MLPClassifier(solver='adam', activation="tanh", hidden_layer_sizes=(90, 30), random_state=1)
    clf.fit(reducedx, trainy)
    if(testx is not None):
        disp = plot_confusion_matrix(clf, testx, testy, normalize='true')
        disp.ax_.set_title("NeuralNets ConfusionMatrix")
        plt.savefig("NeuralNets ConfusionMatrix({alg}).png".format(alg=alg))
        plt.clf()
    train_sizes = [2000, 3000, 4000, 5000, 6000, 7000, 8344]
    trainsizes, train_scores, validation_scores = learning_curve(
        estimator=MLPClassifier(solver='sgd', activation="tanh", hidden_layer_sizes=(90, 30,), random_state=1),
        X=reducedx, y=trainy, scoring='f1_weighted', train_sizes=train_sizes,
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
    plt.savefig("NeuralNetsLearningCurve-Avila({alg}).png".format(alg=alg))
    plt.clf()

if __name__=="__main__":
    trainx, trainy= get_data()
    trainx_df = pd.DataFrame(trainx)
    trainy_df = pd.DataFrame(trainy)
    #fig = sb.pairplot(trainx_df)
    #fig.savefig("Pairplot-Avila(all features).png")
    plot_original_clusters(trainx_df, trainy_df)
    labels = clustering_kmeans(trainx_df, trainy, False,None,2)
    lb_df = pd.DataFrame(labels)
    labels_em = clustering_em(trainx_df, trainy, False,None,2)
    lb_em_df = pd.DataFrame(labels_em)
    reduced = dim_red_pca(trainx_df, trainy)
    trainrx, testrx, trainry, testry = train_test_split(reduced, trainy, train_size=0.9)
    train_NN(trainrx, trainry, "pca", testrx, testry)
    #print(reduced)
    prinicipal = reduced.iloc[:,:6]
    clustering_kmeans(prinicipal, trainy, True, 'pca', 2, [-2,6], [-4,8])
    clustering_em(prinicipal, trainy, True, "pca", 2,[-10,10], [-6,8] )
    reduced = dim_red_ica(trainx_df)
    trainrx, testrx, trainry, testry = train_test_split(reduced, trainy, train_size=0.9)
    train_NN(trainrx,trainry,"ica",testrx,testry)
    clustering_kmeans(reduced, trainy, True,'ica',2,[-0.02, 0.04],[-0.05, 0.15])
    clustering_em(reduced, trainy, True,'ica',2,[-7,15],[-6,3])
    reduced = dim_red_rp(trainx_df)
    trainrx, testrx, trainry, testry = train_test_split(reduced, trainy, train_size=0.9)
    train_NN(trainrx, trainry,"rp", testrx, testry)
    clustering_kmeans(reduced, trainy, True, 'rp', 2, [-10, 30], [-10, 30])
    clustering_em(reduced, trainy, True,'rp',2,[-7,15],[-6,3])
    reduced = dim_red_svd(trainx_df)
    trainrx, testrx, trainry, testry = train_test_split(reduced, trainy, train_size=0.9)
    train_NN(trainrx, trainry,"svd",testrx,testry)
    clustering_kmeans(reduced, trainy, True, 'svd', 2, [-7, 15], [-6, 3])
    clustering_em(reduced, trainy, True,'svd',2,[-7,15],[-6,3])
    new_train = pd.concat([trainx_df,lb_df], axis=1)
    train_NN(new_train, trainy,"kmeans")
    new_train_em = pd.concat([trainx_df, lb_em_df], axis=1)
    train_NN(new_train_em, trainy,"EM")