import numpy as np
import numpy.linalg as LA

A=np.asarray([[5,1],[3,3]])
Lambda, U = LA.eigh(A)

print(A)
print(U)
print(Lambda)

print(np.dot(U,np.transpose(np.conj(U))))
print(np.dot(np.transpose(np.conj(U)),U))
print(LA.inv(U)-np.transpose(np.conj(U)))

print(np.dot(np.dot(U,np.diag(Lambda)),np.transpose(np.conj(U)))-A)


B=np.random.rand(2,3)
U_,S,V=LA.svd(B,full_matrices = 0)
print(B)
print(U_)
print(S)
print(V)

print(np.dot(U_, np.transpose(np.conj(U_))))
print(np.dot(np.transpose(np.conj(U_)), U_))
print(np.dot(V, np.transpose(np.conj(V))))
print(np.dot(np.transpose(np.conj(V)), V))