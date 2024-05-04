#Kütüphaneleri yükle

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap



#Dosyayi Yukle

os.chdir('C:\\Users\\VOLKAN\\Desktop\\leafs')

dataset = pd.read_csv('C:\\Users\\VOLKAN\\Desktop\\leafs\\leaf.csv')



#Giris cikis belirle

X = dataset.iloc[:, [4,5]].values

y = dataset.iloc[:, 6].values.astype('int')



#Egitim ve test verilerini ayir

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



#Standardizasyon

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)



#Destek Vektör Makinesi

classifier = SVC(kernel='rbf', random_state = 0)

classifier.fit(X_train, y_train)



#Başarıyı belirle

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy score %.3f" %metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                 c = ListedColormap(('yellow', 'green'))(i), label = j)
     
plt.title('DVM (Eğitim Seti)')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
