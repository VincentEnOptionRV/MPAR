import numpy as np
from numpy.linalg import norm
Vn = np.array([0.]*5)
Vnp1 = np.array([0.]*5)
rs = [0,5,100,500,3]
gamma = 1/2
n = 0
print(f"{n = }, {Vn = }")
while(n == 0 or norm(Vnp1 -Vn) > 1):
    Vn = Vnp1.copy()
    Vnp1[0] = gamma*max(0.5*Vn[1] + 0.5*Vn[2], 0.1*Vn[3] + 0.9*Vn[4])
    for i in range(1,5):
        Vnp1[i] = rs[i] + gamma*Vn[0]
    n = n+1
    print(f"{n = }, {Vn = }, {Vnp1 = }, {norm(Vnp1 -Vn) = }")