# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


X=train.drop('ACTION',axis=1)
Y=train.ACTION

#Imbalanced dataset
np.sum(Y==1)
np.sum(Y==0)

#Balancing Imbalanced data
from imblearn.over_sampling import SMOTE
import imblearn
oversample=SMOTE()
X,Y=oversample.fit_resample(X,Y)
counter=Counter(Y)
    

#OneHotEncoder
from sklearn import preprocessing
le=preprocessing.OneHotEncoder()
le.fit(X)
onehotlabels = le.transform(X) 

#FeatureSelection
from sklearn.feature_selection import SelectKBest, chi2
select=SelectKBest(chi2,k=7)
select.fit(X,Y)
modifiedData=select.transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

#Classifier
classifier= RandomForestClassifier()
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,Y_test)

accuracy=np.trace(cm)/np.sum(cm)

