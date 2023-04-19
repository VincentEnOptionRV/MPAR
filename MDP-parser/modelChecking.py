from mdp import MDP
import numpy as np
from scipy.sparse.csgraph import connected_components
from numpy.linalg import solve
from numpy.linalg import lstsq
from scipy.optimize import linprog


def buildAdversary(mdp:MDP):
    print("Build Adversary")

    adversary = [-1] * len(mdp.states)

    for i_state in range(len(mdp.states)):
        # print(mdp.states[i_state])
        # print(mdp.accessible[i_state])
        i_possibleActions = mdp.accessible[i_state]
        possibleAction = [mdp.actions[i_action] for i_action in i_possibleActions]

        if len(i_possibleActions) > 1:
            print(f'Choose an action from {possibleAction} for state {mdp.states[i_state]} :')
            action = ""
            while action not in possibleAction:
                print("Enter an action :")
                action = input()

            i_action = np.where(mdp.actions == action)[0][0]
            adversary[i_state] = i_action
        else:
            adversary[i_state] = i_possibleActions[0]
            # print(mdp)
    print("Adversary Built")
    return adversary


def modelChecking(mdp:MDP, adversary, states, n=None):
    i_states = [np.where(mdp.states == state)[0][0] for state in states]
                        
    mc = np.array([mdp.P[adversary[i_state]][i_state] for i_state in range(len(mdp.states))])

    S0s, S1s = zip(*[_calcS0S1(mc, i_states[i]) for i in range(len(states))])
    S0, S1 = S0s[0].intersection(*S0s), S1s[0].union(*S1s)

    if n is not None: S1 = set(i_states)

    S_unknown = list(set(range(len(mdp.states))).difference(S0, S1))

    A = np.zeros((len(S_unknown), len(S_unknown)))

    for i in range(len(S_unknown)):
        for j in range(len(S_unknown)):
            A[i][j] = mc[S_unknown[i]][S_unknown[j]]
            
    
    b = np.zeros(len(S_unknown))
    for i in range(len(S_unknown)):
        for j in S1:
            b[i] += mc[S_unknown[i]][j]

    if n is None:
        try: proba = solve(np.eye(len(S_unknown)) - A, b)
        except: proba = lstsq(np.eye(len(S_unknown)) - A, b, rcond=None)[0]

        if len(states) > 1: print(f'Prop = <>({" v ".join(states)})')
        else: print(f'Prop = <>{states[0]}')
    else:
        proba = np.zeros(len(S_unknown))
        for _ in range(n):
            proba = np.matmul(A, proba) + b

        if len(states) > 1: print(f'Prop = <> <={n} ({" v ".join(states)})')
        else: print(f'Prop = <> <={n} {states[0]}')

    maxStateLen = max([len(s) for s in mdp.states])
    for i_state in range(len(mdp.states)):
        if i_state in S0:
            print(f'P({mdp.states[i_state]:{maxStateLen}}|= Prop) =   0.0%')
        elif i_state in S1:
            print(f'P({mdp.states[i_state]:{maxStateLen}}|= Prop) = 100.0%')
        else:
            print(f'P({mdp.states[i_state]:{maxStateLen}}|= Prop) = {proba[S_unknown.index(i_state)]:6.1%}')


def modelCheckingMaxMin(mdp:MDP, states):
    i_states = [np.where(mdp.states == state)[0][0] for state in states]
    
    A = np.zeros((len(mdp.states) * len(mdp.actions), len(mdp.states)))
    b = np.zeros(len(mdp.states) * len(mdp.actions))

    for i_s in range(len(mdp.states)):
        for i_a in range(len(mdp.actions)):
            for j_s in range(len(mdp.states)):
                if i_s == j_s: aij = mdp.P[i_a][i_s][j_s] - 1
                else: aij = mdp.P[i_a][i_s][j_s]
                A[i_s * len(mdp.actions) + i_a][j_s] = aij

    for i_s in range(len(mdp.states)):
        for i_a in range(len(mdp.actions)):
            for j_s in i_states:
                b[i_s * len(mdp.actions) + i_a] -= mdp.P[i_a][i_s][j_s]

    for i_s in i_states:
        for j_s in range(len(mdp.states)):
            for i_a in range(len(mdp.actions)):
                A[i_s * len(mdp.actions) + i_a][j_s] = 0

    for i_s in i_states:
        for i_a in range(len(mdp.actions)):
            b[i_s * len(mdp.actions) + i_a] = 0

    res = linprog(c=np.ones(len(mdp.states)), A_ub=A, b_ub=b, bounds=(0,1))

    if len(states) > 1: print(f'Prop = <>({" v ".join(states)})')
    else: print(f'Prop = <>{states[0]}')

    maxStateLen = max([len(s) for s in mdp.states])
    for i_state in range(len(mdp.states)):
        if i_state in i_states:
            print(f'Pmax({mdp.states[i_state]:{maxStateLen}}|= Prop) = 100.0%')
        else:
            print(f'Pmax({mdp.states[i_state]:{maxStateLen}}|= Prop) = {res.x[i_state]:6.1%}')


def _calcS0S1(mc, i_state):
    S1 = {i_state}
    S0 = set()

    for i in range(len(mc)):
        if mc[i, i_state] == 1: S1.add(i)
        if i != i_state and mc[i,i] == 1: S0.add(i)

    return S0, S1