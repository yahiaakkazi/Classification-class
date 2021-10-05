import random
def generateur(a,b,position,x_max,y_max,nb_points):
    abscicces=np.array([random.uniform(0,x_max) for i in range(0,nb_points) ])
    if position=="haut":
        #abscicces=np.array([random.uniform(0,100) for i in range(0,100) ])
        ordonnes=[]
        for i in abscicces:
            ordonnes=ordonnes+[a*i+b+random.uniform(0,y_max)]
    else:
        #abscicces=np.array([random.uniform(0,100) for i in range(0,100) ])
        ordonnes=[]
        for i in abscicces:
            ordonnes=ordonnes+[a*i+b-random.uniform(0,y_max)]
    Y=np.array([random.uniform(0,100) for i in range(0,10)])
    T=np.array([random.uniform(3*i,3*i+69) for i in Y])
    return abscicces,np.array(ordonnes),Y,T


import matplotlib.pyplot as plt

I=generateur(3,70,"haut",100,100,100)
U=generateur(3,0,"bas",100,100,100)

X=[[i,j] for (i,j) in zip(U[0],U[1])]
Y=[[i,j] for (i,j) in zip(I[0],I[1])]
données_parfaites=X+Y
Z=[[i,j] for (i,j) in zip(I[2],I[3])]
données_non_parfaites=X+Y+Z
L=[0 for i in range(0,100)]+[1 for i in range(0,100)]
M=L+[random.randint(0,1) for i in range(0,10)]
plt.plot(I[0], I[1], 'o', color='black')
plt.plot(U[0], U[1], 'o', color='red')
#plt.plot(I[2],I[3],'o',color='blue')


from sklearn import svm
clf=svm.SVC(C=0.000005,kernel="linear")
clf.fit(X+Y,L)

clf.fit(données_non_parfaites,M)


plt.plot(clf.support_vectors_[:,0],clf.support_vectors_[:,1],'o', color='blue')
plt.plot(clf.coef_[0][0],clf.intercept_[0],'o',color='green')
plt.show()




