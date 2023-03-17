from mdp import MDP
import random
import numpy as np

def qlearning(mdp:MDP, gamma, Ttot):
    n_etats = len(mdp.states)
    n_actions = len(mdp.actions)
    Qt = np.zeros((n_etats, n_actions))
    Qtp1 = np.zeros((n_etats, n_actions))
    alpha = np.ones((n_etats, n_actions))
    st = 0
    stp1 = 0

    for t in range(Ttot):
        st = stp1
        at = np.random.choice(a=mdp.accessible[st], size=1)[0]
        stp1 = np.random.choice(a=mdp.P.shape[2], size=1, p=mdp.P[at][st])[0]
        rt = mdp.rewards[st]
        Qt = Qtp1.copy()
        dt = rt + gamma*np.max(Qt[stp1]) - Qt[st,at]
        Qtp1[st,at] = Qt[st,at] + (1/alpha[st,at])*dt
        alpha[st,at]+=1
    
    # At this point, Qtp1 is supposedly close to Q*
    
    bestAdversary = np.argmax(a=Qtp1, axis=1)
    # bestAdversaryValues = np.max(a=Qtp1, axis=1) # debug

    return bestAdversary