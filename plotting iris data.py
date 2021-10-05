from sklearn import datasets
iris = datasets.load_iris()

import matplotlib.pyplot as plt

#plt.plot(iris.data[:,3],iris.data[:,2], 'ro')
#plt.show()


#plt.scatter(iris.data[:,1][0:50], iris.target[0:50], c = 'red', marker='o')
#plt.scatter(iris.data[:,3][0:50], iris.target[50:100], c = 'blue', marker="+")

plt.scatter(iris.data[:,0], iris.target[:,1], c =iris.target, marker="+")

#plt.plot(iris.data[:,1],iris.data[:,3], 'ro')
plt.show()

