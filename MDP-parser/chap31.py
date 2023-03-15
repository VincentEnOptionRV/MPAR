import numpy as np
import random
Qt = np.zeros((5,3))
Qtp1 = np.zeros((5,3))
rs = [0,5,100,500,3]

gamma = 0.5
st = 0
stp1 = 0

def chooseAction(st):
    if st == 0:
        return random.randint(1,2)
    else:
        return 0

def simulate(st,at):
    if st == 0:
        if at == 1:
            return random.randint(1,2)
        else:
            tirage = random.randint(1,10)
            if tirage == 1:
                return 4
            else:
                return 3
    else:
        return 0

Ttot = 1000
for t in range(Ttot):
    st = stp1
    at = chooseAction(st)
    stp1 = simulate(st,at)
    rt = rs[stp1]
    Qt = Qtp1.copy()
    dt = rt + gamma*np.max(Qt[stp1]) - Qt[st,at]
    Qtp1[st,at] = Qt[st,at] + 1/(t+1)*dt
print(f"{Qtp1 = }")