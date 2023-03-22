from mdp import MDP
import random
import numpy as np
from tqdm import tqdm

def qlearning(mdp:MDP, gamma, Ttot, Q0=None, start=None):
    n_etats = len(mdp.states)
    n_actions = len(mdp.actions)

    if Q0 is None:
        Qt = np.zeros((n_etats, n_actions))
        Qtp1 = np.zeros((n_etats, n_actions))
    else:
        Qt = Q0
        Qtp1 = Q0
    alpha = np.ones((n_etats, n_actions))
    if start is None:
        st = 0
        stp1 = 0
    else:
        st = start
        stp1 = start

    infboucle = []
    for t in tqdm(range(Ttot)):
        st = stp1
        if st == stp1:
            proba_boucler = [mdp.P[act][st][st] for act in mdp.accessible[st]]
            if np.mean(proba_boucler) == 1:   # Si on boucle à l'infini
                infboucle.append(st)
                st = np.random.choice(a=len(mdp.states), size=1)[0] # Alors on se TP
        at = np.random.choice(a=mdp.accessible[st], size=1)[0]
        stp1 = np.random.choice(a=mdp.P.shape[2], size=1, p=mdp.P[at][st])[0]
        rt = mdp.rewards[st]
        Qt = Qtp1.copy()
        dt = rt + gamma*np.max(Qt[stp1]) - Qt[st,at]
        Qtp1[st,at] = Qt[st,at] + (1/alpha[st,at])*dt
        alpha[st,at]+=1
    
    # At this point, Qtp1 is supposedly close to Q*
    
    bestAdversary = np.argmax(a=Qtp1, axis=1)
    bestAdversaryValues = np.max(a=Qtp1, axis=1)

    return bestAdversary, bestAdversaryValues, Qtp1, infboucle