#SVD: single value decomposition
import numpy as np
from numpy.linalg import svd


A=np.array([[1,5,2],[4,3,0],[8,0,1]])

print(A)

U, s, VT = svd(A)

print(U)
print(s)
print(VT)

S=np.diag(s)
print(S)
print(U.dot(S.dot(VT)))
print(U.dot(S.dot(VT)).round())
U.dot(S.dot(VT)).round()==A

B=np.array([[1,5,2],[4,3,0],[8,0,1],[8,0,3]])

I,d,CT=svd(B)
D=np.zeros(np.shape(B))
D[:len(d),:len(d)]=np.diag(d)

I.dot(D.dot(CT)).round()==B
