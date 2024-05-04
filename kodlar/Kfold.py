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

from sklearn.model_selection import KFold

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



siniflandiricilar=[KNeighborsClassifier(n_neighbors=3), LogisticRegression(random_state=0), GaussianNB(), DecisionTreeClassifier(), SVC(), RandomForestClassifier(n_estimators=50,)]



basari=list()

kSayisi = 10

kf = KFold(n_splits=kSayisi)   

fSkor = list()

for i in range(6):   

    toplamBasari = 0;

    toplamfSkor = 0;

    for egitim_index, test_index in kf.split(giris_verileri):

        #Standardizasyon

        scaler = preprocessing.StandardScaler()

        stdGiris = scaler.fit_transform(giris_verileri.iloc[egitim_index,:])

        stdTest = scaler.transform(giris_verileri.iloc[test_index,:])        

        siniflandiricilar[i].fit(stdGiris, cikis[egitim_index] )        

        cikis_tahmin = siniflandiricilar[i].predict(stdTest)

        toplamBasari += (accuracy_score(cikis[test_index], cikis_tahmin))        

        toplamfSkor += ( f1_score(cikis[test_index], cikis_tahmin, labels=None, pos_label=1, average='weighted', sample_weight=None))     
        
    print(confusion_matrix(cikis[test_index], cikis_tahmin))

    basari.append(toplamBasari/kSayisi)

    fSkor.append(toplamfSkor/kSayisi)
    
    print(classification_report(cikis[test_index], cikis_tahmin))
    
    print(accuracy_score(cikis[test_index], cikis_tahmin))