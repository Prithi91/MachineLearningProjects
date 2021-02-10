import csv
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',None)
import matplotlib
matplotlib.use('Agg')
import re
import math
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score, silhouette_score
from sklearn.metrics import mean_squared_error
import seaborn as sb
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis

def get_data():
    dataset_path = "breastcancerproteomes/77_cancer_proteomes_CPTAC_itraq.csv"
    clinical_info = "breastcancerproteomes/clinical_data_breast_cancer.csv"
    pam50_proteins = "breastcancerproteomes/PAM50_proteins.csv"
    data = pd.read_csv(dataset_path, header=0, index_col=0)
    clinical = pd.read_csv(clinical_info, header=0,
                           index_col=0)
    pam50 = pd.read_csv(pam50_proteins, header=0)
    data.drop(['gene_symbol', 'gene_name'], axis=1, inplace=True)
    data = data.iloc[:, :-3]
    data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]', x)[0]) if bool(re.search("TCGA", x)) is True else x,
                inplace=True)
    data = data.transpose()
    clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index], :]
    merged = data.merge(clinical, left_index=True, right_index=True)
    processed_numerical = merged.loc[:, [x for x in merged.columns if bool(re.search("NP_|XP_", x)) == True]]
    processed_numerical_p50 = processed_numerical.iloc[:, processed_numerical.columns.isin(pam50['RefSeqProteinID'])]
    processed_numerical_p50 = processed_numerical_p50.fillna(processed_numerical_p50.mean())
    return processed_numerical_p50, merged

def clustering_kmeans(processed_numerical_p50, merged, isreduced=False, alg='pca', finalclusters=2):
    #processed_numerical_p50, merged = get_data()
    n_clusters = range(2, 15)
    scores = []
    homogeneity_scores = []
    for i in n_clusters:
        kclusters = KMeans(n_clusters=i, n_jobs=4)
        kclusters.fit(processed_numerical_p50)
        scores.append(silhouette_score(processed_numerical_p50, kclusters.labels_))
        #if(isreduced==False):
        homogeneity_scores.append(homogeneity_score(merged['PAM50 mRNA'], kclusters.labels_))
    plt.xlabel('K values')
    plt.ylabel('Scores')
    plt.title('Scores(Kmeans Clustering)')
    plt.plot(n_clusters, scores, label='Silhouette Scores')
    if(isreduced==False):
        plt.plot(n_clusters, homogeneity_scores, label='Homogeneity Scores')
        plt.legend()
        plt.savefig('Silhouette Method-breastcancer.png')
    else:
        plt.plot(n_clusters, homogeneity_scores, label='Homogeneity Scores')
        plt.legend()
        plt.savefig("Silhouette Method-breastcancer(DimRed{alg}_{nc}).png".format(alg=alg, nc=finalclusters))
    plt.clf()
    if (isreduced == False):
        kclusters_final = KMeans(n_clusters=3, n_jobs=4)
        kclusters_final = kclusters_final.fit(processed_numerical_p50)
        #print(kclusters_final.labels_)
        processed_numerical_p50['kmeans_cluster'] = kclusters_final.labels_
        labels = pd.DataFrame(kclusters_final.labels_, columns=['Clusters'])
        labels.sort_values('Clusters', axis=0, inplace=True)
        labels['pam50RNA'] = list(merged['PAM50 mRNA'])
        labels = labels.replace("Basal-like",0)
        labels = labels.replace("HER2-enriched",1)
        labels = labels.replace("Luminal A",2)
        labels = labels.replace("Luminal B",3)
        labels.index.name = 'Clusters'
        ax= plt.axes()
        sb.heatmap(labels, ax=ax)
        ax.set_title("Kmeans- Clustering")
        #fig = map.get_figure()
        plt.savefig('Heatmap_BreastCancer(clusteringKM).png', dpi=400)
        plt.clf()
        processed_numerical_p50.sort_values('kmeans_cluster', axis=0, inplace=True)
        processed_numerical_p50.index.name = 'Patient'
        #print(processed_numerical_p50)
        map = sb.heatmap(processed_numerical_p50)
        fig = map.get_figure()
        fig.savefig('Heatmap_BreastCancer.png', dpi=400)
        plt.clf()
    else:
        kclusters_final = KMeans(n_clusters=finalclusters, n_jobs=4)
        kclusters_final = kclusters_final.fit(processed_numerical_p50)
        # print(kclusters_final.labels_)
        processed_numerical_p50['kmeans_cluster'] = kclusters_final.labels_
        labels = pd.DataFrame(kclusters_final.labels_, columns=['Clusters'])
        labels.sort_values('Clusters', axis=0, inplace=True)
        labels['pam50RNA'] = list(merged['PAM50 mRNA'])
        labels = labels.replace("Basal-like", 0)
        labels = labels.replace("HER2-enriched", 1)
        labels = labels.replace("Luminal A", 2)
        labels = labels.replace("Luminal B", 3)
        labels.index.name = 'Clusters'
        ax = plt.axes()
        sb.heatmap(labels, ax=ax)
        ax.set_title("Kmeans- Clustering")
        # fig = map.get_figure()
        plt.savefig("Heatmap_BreastCancer(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters), dpi=400)
        plt.clf()
        #processed_numerical_p50.sort_values('kmeans_cluster', axis=0, inplace=True)
        #processed_numerical_p50.index.name = 'Patient'
        #map = sb.heatmap(processed_numerical_p50)
        #fig = map.get_figure()
        #fig.savefig("Heatmap_BreastCancer(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters), dpi=400)
        values = [0,1,2,3]
        #colors = cm.rainbow(np.linspace(0,1,4))
        colors=['blue','red','green','black']
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Scatter Plot - proteome based clustering')
        for v,color in zip(values,colors):
            indices = processed_numerical_p50['kmeans_cluster'] == v
            ax.scatter(processed_numerical_p50.loc[indices, 0], processed_numerical_p50.loc[indices, 1], c=color)
        plt.savefig("Scatter Plot- {alg}(Clustered)_{nc}".format(alg=alg,nc=finalclusters))
        plt.clf()


def clustering_em(processed_numerical_p50, merged, isreduced=False, alg='pca', finalclusters=2):
    #processed_numerical_p50, merged = get_data()
    n_clusters = range(2, 15)
    scores = []
    homogeneity_scores = []
    for i in n_clusters:
        gmm = GaussianMixture(n_components=i, covariance_type='full')
        kclusters = gmm.fit(processed_numerical_p50)
        y_pred = gmm.predict(processed_numerical_p50)
        scores.append(silhouette_score(processed_numerical_p50, y_pred))
        homogeneity_scores.append(homogeneity_score(merged['PAM50 mRNA'], y_pred))
    plt.xlabel('K values')
    plt.ylabel('Scores')
    plt.title('Silhouette Method - EM clustering')
    plt.plot(n_clusters, scores, label='Silhouette Score')
    if(isreduced==False):
        plt.plot(n_clusters, homogeneity_scores,label='Homogeneity Scores')
        plt.legend()
        plt.savefig('Silhouette Method(EM)-breastcancer.png')
    else:
        plt.plot(n_clusters, homogeneity_scores,label='Homogeneity Scores')
        plt.legend()
        plt.savefig("Silhouette Method(EM)-breastcancer(DimRed{alg}_{nc}).png".format(alg=alg, nc=finalclusters))
    plt.clf()
    if(isreduced==False):
        gmm_final = GaussianMixture(n_components=3, covariance_type='diag')
        model = gmm_final.fit(processed_numerical_p50)
        y_pred = gmm_final.predict(processed_numerical_p50)
        processed_numerical_p50['kmeans_cluster'] = y_pred
        labels = pd.DataFrame(y_pred, columns=['Clusters'])
        labels.sort_values('Clusters', axis=0, inplace=True)
        labels['pam50RNA'] = list(merged['PAM50 mRNA'])
        labels = labels.replace("Basal-like", 0)
        labels = labels.replace("HER2-enriched", 1)
        labels = labels.replace("Luminal A", 2)
        labels = labels.replace("Luminal B", 3)
        labels.index.name = 'Clusters'
        ax = plt.axes()
        sb.heatmap(labels, ax=ax)
        ax.set_title("EM(GMM)- Clustering")
        plt.savefig('Heatmap_BreastCancer(clusteringEM).png', dpi=400)
        plt.clf()
        processed_numerical_p50.sort_values('kmeans_cluster', axis=0, inplace=True)
        processed_numerical_p50.index.name = 'Patient'
        #print(processed_numerical_p50)
        map = sb.heatmap(processed_numerical_p50)
        fig = map.get_figure()
        fig.savefig('Heatmap(EM)_BreastCancer.png', dpi=400)
        plt.clf()
    else:
        gmm_final = GaussianMixture(n_components=finalclusters, covariance_type='diag')
        model = gmm_final.fit(processed_numerical_p50)
        y_pred = gmm_final.predict(processed_numerical_p50)
        processed_numerical_p50['kmeans_cluster'] = y_pred
        labels = pd.DataFrame(y_pred, columns=['Clusters'])
        labels.sort_values('Clusters', axis=0, inplace=True)
        labels['pam50RNA'] = list(merged['PAM50 mRNA'])
        labels = labels.replace("Basal-like", 0)
        labels = labels.replace("HER2-enriched", 1)
        labels = labels.replace("Luminal A", 2)
        labels = labels.replace("Luminal B", 3)
        labels.index.name = 'Clusters'
        ax = plt.axes()
        sb.heatmap(labels, ax=ax)
        ax.set_title("EM(GMM)- Clustering")
        plt.savefig("Heatmap(EM)_BreastCancer(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters), dpi=400)
        plt.clf()
        # processed_numerical_p50.sort_values('kmeans_cluster', axis=0, inplace=True)
        # processed_numerical_p50.index.name = 'Patient'
        # # print(processed_numerical_p50)
        # map = sb.heatmap(processed_numerical_p50)
        # fig = map.get_figure()
        # fig.savefig("Heatmap(EM)_BreastCancer(DimRed{alg}_{nc}).png".format(alg=alg,nc=finalclusters), dpi=400)
        # plt.clf()
        values = [0, 1, 2, 3]
        colors = ['blue', 'red', 'green', 'black']
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Scatter Plot - proteome based clustering')
        for v, color in zip(values, colors):
            indices = processed_numerical_p50['kmeans_cluster'] == v
            ax.scatter(processed_numerical_p50.loc[indices, 0], processed_numerical_p50.loc[indices, 1], c=color)
        plt.savefig("Scatter Plot(EM)- {alg}(Clustered)_{nc}".format(alg=alg, nc=finalclusters))
        plt.clf()

def dim_red_pca(processed_numerical_p50, merged):
    #print(processed_numerical_p50.shape)
    processed_numerical_p50 = StandardScaler().fit_transform(processed_numerical_p50)
    pca = PCA(n_components=5)
    components = pca.fit_transform(processed_numerical_p50)
    # features = range(pca.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_)
    # #plt.bar(features, pca.explained_variance_)
    # plt.title("Distribution of variance among the components")
    # plt.xlabel('PCA features')
    # plt.ylabel('Variance %')
    # plt.xticks(features)
    # plt.savefig("pca_variance.png")
    # plt.clf()
    pc = pd.DataFrame(components)
    pc_final = pc.iloc[:,:2]
    df_scatter = pd.DataFrame(data=pc_final)
    df_scatter.rename(columns={0:"Feature1",1:"Feature2"}, inplace=True)
    #print(df_scatter)
    target = list(merged['PAM50 mRNA'])
    target_df = pd.DataFrame(data=target)
    #print(target_df)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    #print(df_toplot)
    values = ['Basal-like','HER2-enriched','Luminal A','Luminal B']
    colors = ['blue', 'red', 'green', 'black']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Scatter Plot - Clustering based on mRNA')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
    plt.savefig("Scatter Plot- pca(Original)")
    plt.clf()
    # comp = range(5, 42)
    # rmsevals = []
    # for i in comp:
    #     pca = PCA(n_components=i)
    #     reduced = pca.fit_transform(processed_numerical_p50)
    #     #inv = np.linalg.pinv(svd.components_)
    #     reconstructed = pca.inverse_transform(reduced)
    #     rmse = math.sqrt(mean_squared_error(processed_numerical_p50, reconstructed))
    #     rmsevals.append(rmse)
    # plt.plot(comp, rmsevals)
    # plt.xlabel('Number of Components')
    # plt.ylabel('RMSE(Reconstruction error)')
    # plt.title('Reconstruction error as a function of number of components')
    # plt.savefig('RMSE_pca.png')
    # plt.clf()
    return pc

def dim_red_ica(processed_numerical_p50, merged):
    processed_numerical_p50 = StandardScaler().fit_transform(processed_numerical_p50)
    #print(processed_numerical_p50.shape)
    # n= range(1,42)
    # mean_kurt_scores = []
    # for i in n:
    #     ica = FastICA(n_components=i, whiten=True,tol=0.1)
    #     reduced = ica.fit_transform(processed_numerical_p50)
    #     # kurt=[]
    #     # for j in range(i):
    #     #     kurt.append(kurtosis(reduced[j]))
    #     #mean_kurt_scores.append(abs(kurtosis(reduced[i])))
    #     mean_kurt_scores.append(np.mean(kurtosis(reduced)))
    # plt.plot(n, mean_kurt_scores)
    # plt.title("Kurtosis as a function of number of components")
    # plt.xlabel("Number of components")
    # plt.ylabel("Mean Kurtosis")
    # plt.savefig("ica_kurtosis(Breastcancer).png")
    # plt.clf()
    # rmsevals=[]
    # for i in n:
    #     ica = FastICA(n_components=i,whiten=True,tol=0.1)
    #     reduced = ica.fit_transform(processed_numerical_p50)
    #     #inv = np.linalg.pinv(svd.components_)
    #     reconstructed = ica.inverse_transform(reduced)
    #     rmse = math.sqrt(mean_squared_error(processed_numerical_p50, reconstructed))
    #     rmsevals.append(rmse)
    # plt.plot(n, rmsevals)
    # plt.xlabel('Number of Components')
    # plt.ylabel('RMSE(Reconstruction error)')
    # plt.title('Reconstruction error as a function of number of components')
    # plt.savefig('RMSE_ica(Breastcancer).png')
    # plt.clf()
    ica = FastICA(n_components=25)
    reduced = ica.fit_transform(processed_numerical_p50)
    ic = pd.DataFrame(reduced)
    #print(ic)
    pc_final = ic.iloc[:, :2]
    df_scatter = pd.DataFrame(data=pc_final)
    df_scatter.rename(columns={0: "Feature1", 1: "Feature2"}, inplace=True)
    target = list(merged['PAM50 mRNA'])
    target_df = pd.DataFrame(data=target)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B']
    colors = ['blue', 'red', 'green', 'black']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Independent Component 1')
    ax.set_ylabel('Independent Component 2')
    ax.set_title('Scatter Plot(ICA) - Clustering based on mRNA')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
    plt.savefig("Scatter Plot- ICA(Original)")
    plt.clf()
    return ic

def dim_red_rp(processed_numerical_p50):
    comp = range(5,42)
    rmsevals=[]
    for i in comp:
        rp = GaussianRandomProjection(n_components=i,eps=0.001)
        reduced = rp.fit_transform(processed_numerical_p50)
        inv = np.linalg.pinv(rp.components_)
        reconstructed = np.dot(reduced, inv.T)
        rmse = math.sqrt(mean_squared_error(processed_numerical_p50, reconstructed))
        rmsevals.append(rmse)
    plt.plot(comp,rmsevals, label='Average of error over 10 runs')
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE(Reconstruction error)')
    plt.title('Construction error as a function of number of components')
    plt.legend()
    plt.savefig('RMSE_rp(Breastcancer).png')
    plt.clf()
    rp = GaussianRandomProjection(random_state=0, n_components=40)
    reduced = rp.fit_transform(processed_numerical_p50)
    rp_x = pd.DataFrame(reduced)
    pc_final = rp_x.iloc[:, :2]
    df_scatter = pd.DataFrame(data=pc_final)
    df_scatter.rename(columns={0: "Feature1", 1: "Feature2"}, inplace=True)
    target = list(merged['PAM50 mRNA'])
    target_df = pd.DataFrame(data=target)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B']
    colors = ['blue', 'red', 'green', 'black']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Random Component 1')
    ax.set_ylabel('Random Component 2')
    ax.set_title('Scatter Plot(RP) - Clustering based on mRNA')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
    plt.savefig("Scatter Plot- RP(Original)")
    plt.clf()
    return rp_x

def dim_red_svd(processed_numerical_p50):
    comp = range(5,42)
    rmsevals=[]
    for i in comp:
        svd = TruncatedSVD(random_state=0, n_components=i)
        reduced = svd.fit_transform(processed_numerical_p50)
        inv = np.linalg.pinv(svd.components_)
        reconstructed = np.dot(reduced, inv.T)
        rmse = math.sqrt(mean_squared_error(processed_numerical_p50, reconstructed))
        rmsevals.append(rmse)
    plt.plot(comp, rmsevals)
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE(Reconstruction error)')
    plt.title('Construction error as a function of number of components')
    plt.savefig('RMSE_svd(Breastcancer).png')
    plt.clf()
    svd = TruncatedSVD(random_state=0, n_components=30)
    reduced = svd.fit_transform(processed_numerical_p50)
    svd_x = pd.DataFrame(reduced)
    pc_final = svd_x.iloc[:, :2]
    df_scatter = pd.DataFrame(data=pc_final)
    df_scatter.rename(columns={0: "Feature1", 1: "Feature2"}, inplace=True)
    target = list(merged['PAM50 mRNA'])
    target_df = pd.DataFrame(data=target)
    df_toplot = pd.concat([df_scatter, target_df], axis=1)
    values = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B']
    colors = ['blue', 'red', 'green', 'black']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Scatter Plot(SVD) - Clustering based on mRNA')
    for v, color in zip(values, colors):
        indices = df_toplot[0] == v
        ax.scatter(df_toplot.loc[indices, 'Feature1'], df_toplot.loc[indices, 'Feature2'], c=color)
    plt.savefig("Scatter Plot- SVD(Original)")
    plt.clf()
    return svd_x

if __name__=="__main__":
    processed_numerical_p50, merged = get_data()
    clustering_kmeans(processed_numerical_p50, merged)
    clustering_em(processed_numerical_p50, merged)
    x_reduced = dim_red_pca(processed_numerical_p50, merged)
    x_principal = x_reduced.iloc[:,:6]
    clustering_kmeans(x_principal, merged, True,"pca",3)
    clustering_em(x_principal,merged,True,"pca",3)
    # clustering_kmeans(x_principal, merged, True,"pca",3)
    x_independent = dim_red_ica(processed_numerical_p50, merged)
    x_independent = x_reduced.iloc[:,:4]
    clustering_kmeans(x_independent, merged, True,"ica",3)
    clustering_em(x_independent, merged, True, "ica", 3)
    x_reduced = dim_red_rp(processed_numerical_p50)
    clustering_kmeans(x_reduced, merged, True, "rp", 3)
    clustering_em(x_reduced, merged, True, "rp", 3)
    x_reduced = dim_red_svd(processed_numerical_p50)
    clustering_kmeans(x_reduced,merged,True,"svd",3)
    clustering_em(x_reduced, merged, True, "svd",4)