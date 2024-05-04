# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import LeaveOneOut

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

#Dosyayi Yukle

veri = pd.read_csv('C:\\Users\\VOLKAN\\Desktop\\leafs\\leaf.csv')

ozellik_sayisi = 16



#giris cikis belirle

giris_verileri = veri.iloc[:,1:ozellik_sayisi+1]

cikis = veri.iloc[:,-1].astype('int')



#Egitim ve test verilerini ayir

egitim_giris, test_giris,egitim_cikis, test_cikis = train_test_split(giris_verileri,cikis, test_size=0.3, random_state=0)



#Standardizasyon

scaler = preprocessing.StandardScaler()

stdGiris = scaler.fit_transform(egitim_giris)

stdTest = scaler.transform(test_giris)

siniflandiricilar=[KNeighborsClassifier(n_neighbors=3), LogisticRegression(random_state=0), GaussianNB(), DecisionTreeClassifier(), SVC(), RandomForestClassifier(n_estimators=50,)]



basari=list()



fSkor = list()

for i in range(6):

    siniflandiricilar[i].fit(stdGiris, egitim_cikis)

    cikis_tahmin = siniflandiricilar[i].predict(stdTest)

    print(confusion_matrix(test_cikis, cikis_tahmin))

    basari.append(accuracy_score(test_cikis, cikis_tahmin))

    fSkor.append( f1_score(test_cikis, cikis_tahmin, labels=None, pos_label=1, average='weighted', sample_weight=None))
    
    print(classification_report(test_cikis, cikis_tahmin))
    
    print(accuracy_score(test_cikis, cikis_tahmin))