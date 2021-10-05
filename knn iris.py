from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pandas as pd
iris = datasets.load_iris()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import numpy as np

def model_evaluation(y_pred, y_test):        #works only on one
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n{}".format(confusion))
    print("Precision score: {:.2f}".format(precision_score(y_test, y_pred)))
    print("Recall score: {:.2f}".format(recall_score(y_test, y_pred)))
    print("f1 score: {:.2f}".format(f1_score(y_test, y_pred)))
    print("accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))

X,y=shuffle(iris.data,iris.target)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

X_train_a, X_test_a, y_train_a , y_test_a = X[0:120],X[120:150],y[0:120],y[120:150]
X_train_b, X_test_b, y_train_b , y_test_b = np.concatenate((X[0:30],X[60:150])),X[30:60],np.concatenate((y[0:30],y[60:150])),y[30:60]
X_train_c, X_test_c, y_train_c , y_test_c = np.concatenate((X[0:90],X[120:150])),X[90:120],np.concatenate((y[0:90],y[120:150])),y[90:120]
X_train_d, X_test_d, y_train_d , y_test_d = np.concatenate((X[0:60],X[90:150])),X[60:90],np.concatenate((y[0:60],y[90:150])),y[60:90]

X_train=[X_train_a,X_train_b,X_train_c,X_train_d]
X_test=[X_test_a,X_test_b,X_test_c,X_test_d]
y_train=[y_train_a,y_train_b,y_train_c,y_train_d]
y_test=[y_test_a,y_test_b,y_test_c,y_test_d]

y_train_setosa_others,y_train_versicolor_others,y_train_virginica_others=[],[],[]
y_test_setosa_others,y_test_versicolor_others,y_test_virginica_others=[],[],[]


for i in [y_train_a,y_train_b,y_train_c,y_train_d]:
    y_train_setosa_others=y_train_setosa_others+[np.where(i==2, 1, i)]
    y_train_versicolor_others=y_train_versicolor_others+[np.where(i==0, 2, i)]
    y_train_virginica_others=y_train_virginica_others+[np.where(i==0, 1, i)]

for i in [y_test_a,y_test_b,y_test_c,y_test_d]:
    y_test_setosa_others=y_test_setosa_others+[np.where(i==2, 1, i)]
    y_test_versicolor_others=y_test_versicolor_others+[np.where(i==0, 2, i)]
    y_test_virginica_others=y_test_virginica_others+[np.where(i==0, 1, i)]

y_predict_setosa=[]
y_predict_versicolor=[]
y_predict_virginica=[]
for i in range(0,4):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train[i], y_train_setosa_others[i])
    y_predict_setosa=y_predict_setosa+[classifier.predict(X_test[i])]
    model_evaluation(y_predict_setosa[i],y_test_setosa_others[i])

for i in range(0,4):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train[i], y_train_versicolor_others[i])
    y_predict_versicolor=y_predict_versicolor+[classifier.predict(X_test[i])]
    model_evaluation(y_predict_versicolor[i],y_test_versicolor_others[i])

for i in range(0,4):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train[i], y_train_virginica_others[i])
    y_predict_virginica=y_predict_virginica+[classifier.predict(X_test[i])]
    model_evaluation(y_predict_virginica[i],y_test_virginica_others[i])



