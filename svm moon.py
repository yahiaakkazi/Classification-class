from sklearn.datasets import make_moons
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
def model_evaluation(y_pred, y_test):
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n{}".format(confusion))
    print("Precision score: {:.2f}".format(precision_score(y_test, y_pred)))
    print("Recall score: {:.2f}".format(recall_score(y_test, y_pred)))
    print("f1 score: {:.2f}".format(f1_score(y_test, y_pred)))
    print("accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))


Xmoon,ymoon= make_moons(200,noise=.5, random_state=0)
X,y=shuffle(Xmoon,ymoon)
L=[len(Xmoon)/5*i for i in range(1,6)]
X_train_a, X_test_a, y_train_a , y_test_a = X[0:160],X[160:200],y[0:160],y[160:200]
X_train_b, X_test_b, y_train_b , y_test_b = np.concatenate((X[0:40],X[80:200])),X[40:80],np.concatenate((y[0:40],y[80:200])),y[40:80]
X_train_c, X_test_c, y_train_c , y_test_c = np.concatenate((X[0:120],X[160:200])),X[120:160],np.concatenate((y[0:120],y[160:200])),y[120:160]
X_train_d, X_test_d, y_train_d , y_test_d = np.concatenate((X[0:80],X[120:200])),X[80:120],np.concatenate((y[0:80],y[120:200])),y[80:120]

X_train=[X_train_a,X_train_b,X_train_c,X_train_d]
X_test=[X_test_a,X_test_b,X_test_c,X_test_d]
y_train=[y_train_a,y_train_b,y_train_c,y_train_d]
y_test=[y_test_a,y_test_b,y_test_c,y_test_d]





from sklearn import svm
clf=svm.SVC(C=10000000,kernel="rbf")#gamma=0.1)
clf.fit(X_train_a,y_train_a)
model_evaluation(clf.predict(X_test_a),y_test_a)



plt.scatter(X_train_a[:, 0], X_train_a[:, 1], c=y_train_a, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.show()
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


#gridsearch