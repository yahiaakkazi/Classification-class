from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz, plot_tree, export_text
import numpy as np

iris = datasets.load_iris()

def model_evaluation(y_pred, y_test):
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n{}".format(confusion))
    print("Precision score: {:.2f}".format(precision_score(y_test, y_pred,average="macro")))
    print("Recall score: {:.2f}".format(recall_score(y_test, y_pred,average="macro")))
    print("f1 score: {:.2f}".format(f1_score(y_test, y_pred,average="macro")))
    print("accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))


X,y=shuffle(iris.data,iris.target)

X_train, X_test, y_train , y_test = X[0:120],X[120:150],y[0:120],y[120:150]

classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
model_evaluation(y_predict,y_test)

export_graphviz(classifier,out_file="s.png")
_ = plot_tree(classifier,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)

text_representation = export_text(classifier)
print(text_representation)

