import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time

img=mpimg.imread('images/Heisenberg.png')
plt.imshow(img)
plt.axis('off') 
plt.show()

start_time = time.time()

def SpinOper(ss):
    spin = (ss-1)/2.0
    dz = np.zeros(ss)
    mp = np.zeros(ss-1)

    for i in range(ss):
        dz[i] = spin - i
    for i in range(ss-1):
        mp[i] = np.sqrt((2*spin - i)*(i + 1))
    
    S0 = np.eye(ss)
    Sp = np.diag(mp,1)
    Sm = np.diag(mp,-1)
    Sx = 0.5*(Sp + Sm)
    Sy = -0.5j*(Sp - Sm)
    Sz = np.diag(dz)

    return S0, Sp, Sm, Sx, Sy, Sz

S0, Sp, Sm, Sx, Sy, Sz = SpinOper(2)
print(S0)
print(Sx)
print(Sz)

H2 = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)
H3 = np.kron(H2, S0) + np.kron(S0, H2) 

S, V = LA.eigh(H3)
print(np.sort(S))

end_time = time.time()  
ptime = end_time - start_time 
print(f"calculation: {ptime:.2f} sec")
    
