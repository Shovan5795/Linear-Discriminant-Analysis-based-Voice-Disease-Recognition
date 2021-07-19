# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:21:19 2021

@author: shovon5795
"""
import pandas as pd
dataset = pd.read_csv(r"E:\Research\SAM\Dataset\final_version.csv")

X = dataset.iloc[0:368,0:2770]
y = dataset.iloc[:,-1]


#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_new = scaler.fit_transform(X)

'''
#MinmaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X)
'''

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=None)
X_PCA = lda.fit_transform(X_new, y)

'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_PCA1 = lda.fit_transform(X_new, y)
'''

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_PCA2= pca.fit_transform(X_new)


#K-means++
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', n_init = 10, max_iter=300, random_state=0)
ykmeans = kmeans.fit(X_new)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
ymr = neigh.fit(X_new, y)


#BIRCH
from sklearn.cluster import Birch
mr=Birch(threshold=0.5, branching_factor=25,n_clusters=4)
ymr = mr.fit(X_PCA)

#GMM
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 4, covariance_type = 'spherical', max_iter = 300, init_params = 'random')
ymr = gmm.fit(X_new)


#SVM
from sklearn import svm
#ymr = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_PCA, y)
ymr = svm.SVC(kernel='poly', degree=3, C=1).fit(X_PCA, y)

#DT
from sklearn.tree import DecisionTreeClassifier
ymr = DecisionTreeClassifier().fit(X_PCA, y)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
ymr = LogisticRegression(solver = 'liblinear', multi_class = 'ovr').fit(X_PCA, y)


#GBoost
from sklearn.ensemble import GradientBoostingClassifier
ymr = GradientBoostingClassifier().fit(X_PCA, y)


#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ymr = AdaBoostClassifier(learning_rate = 0.1).fit(X_PCA, y)


#RandomForest
from sklearn.ensemble import RandomForestClassifier
ymr = RandomForestClassifier().fit(X_new, y)


#Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
estimator = []
estimator.append(('K-NN', KNeighborsClassifier(n_neighbors=4)))
estimator.append(('SVM', svm.SVC(kernel='poly', degree=3, C=1)))
estimator.append(('Random Forest', RandomForestClassifier()))
ymr = VotingClassifier(estimators = estimator, voting ='hard').fit(X_new, y)
 


from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, cross_val_score
kf = KFold(n_splits = 10)
skf = StratifiedKFold(n_splits = 10)
shs = ShuffleSplit(n_splits = 10)
sshs = StratifiedShuffleSplit(n_splits = 10)

import datetime
start_time = datetime.datetime.now()
acc1 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=kf, n_jobs=1)
acc2 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=skf, n_jobs=1)
acc3 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=shs, n_jobs=1)
acc4 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=sshs, n_jobs=1)
a1 = acc1.mean()
a2 = acc2.mean()
a3 = acc3.mean()
a4 = acc4.mean()
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = (time_diff.total_seconds() * 1000)/4

'''
#Plotting
import matplotlib.pyplot as plt
plt.scatter(X_PCA2[y == 0, 0], X_PCA2[y == 0, 1], s=100, c='blue', label = 'Healthy')
plt.scatter(X_PCA2[y == 1, 0], X_PCA2[y == 1, 1], s=100, c='green', label = 'Dysphonia')
plt.scatter(X_PCA2[y == 2, 0], X_PCA2[y == 2, 1], s=100, c='red', label = 'Laryngitis')
plt.scatter(X_PCA2[y == 3, 0], X_PCA2[y == 3, 1], s=100, c='cyan', label = 'Reinkes Edema')

plt.title("Scatter Plot Diagram for Different Voice Classes")
plt.xlabel("1st PCA of MFCC Feature")
plt.ylabel("2nd PCA of MFCC Feature")
plt.legend()
plt.show()
'''





