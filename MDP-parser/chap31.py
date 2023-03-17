import numpy as np
import random
Qt = np.zeros((5,3))
Qtp1 = np.zeros((5,3))
rs = [0,100,5,500,3]

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
        elif at == 2:
            tirage = random.randint(1,10)
            if tirage == 1:
                return 3
            else:
                return 4
        else:
            raise Exception("f")
    else:
        return 0

alpha = np.ones((5,3))
Ttot = 10000
for t in range(Ttot):
    st = stp1
    at = chooseAction(st)
    stp1 = simulate(st,at)
    rt = rs[st]
    Qt = Qtp1.copy()
    dt = rt + gamma*np.max(Qt[stp1]) - Qt[st,at]
    Qtp1[st,at] = Qt[st,at] + (1/alpha[st,at])*dt
    alpha[st,at]+=1
print(f"{Qtp1 = }")
bestAdversary = np.argmax(a=Qtp1, axis=1)
print(f"{bestAdversary = }")