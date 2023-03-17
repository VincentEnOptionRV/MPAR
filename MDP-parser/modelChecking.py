from mdp import MDP
import numpy as np
from scipy.sparse.csgraph import connected_components

def buildAdversary(mdp:MDP):
    print("Build Adversary")

    adversary = [-1] * len(mdp.states)

    for i_state in range(len(mdp.states)):
        i_possibleActions = mdp.accessible[i_state]
        possibleAction = [mdp.actions[i_action] for i_action in i_possibleActions]

        if len(i_possibleActions) > 1:
            print(f'Choose an action from {possibleAction} for state {mdp.states[i_state]} :')
            action = ""
            while action not in possibleAction:
                print("\t>> Enter an action :")
                action = input()

            i_action = np.where(mdp.actions == action)[0][0]
            adversary[i_state] = i_action
        else:
            adversary[i_state] = i_possibleActions[0]

    return adversary


def modelChecking(mdp:MDP, adversary, states, n):
    i_states = [np.where(mdp.states == state)[0][0] for state in states]
                        
    mc = np.array([mdp.P[adversary[i_state]][i_state] for i_state in range(len(mdp.states))])
    print(_calcS0S1(mc, i_states[0]))
    S1s = []
    S0s = []

    pass

def _calcS0S1(mc, i_state):
    n_comps, labelsStrong = connected_components(csgraph=mc, directed=True, return_labels=True, connection='strong')
    n_comps, labelsWeak = connected_components(csgraph=mc, directed=True, return_labels=True, connection='weak')

    S1 = {i for i in range(len(labelsStrong)) if labelsStrong[i_state] == labelsStrong[i]}
    S0 = {i for i in range(len(labelsStrong)) if labelsWeak[i_state] != labelsWeak[i]}

    return S0, S1