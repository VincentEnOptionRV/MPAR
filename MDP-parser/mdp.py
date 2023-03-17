from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import numpy as np

        
class gramPrintListener(gramListener):

    def __init__(self):
        pass
        
    def enterStatenoreward(self, ctx):
        print("States: %s" % str([str(x) for x in ctx.ID()]))
        

    def enterStatereward(self, ctx):
        print(f"States: {[str(x) + ':' + str(y) for x, y in zip(ctx.ID(), ctx.INT())]}")


    def enterDefactions(self, ctx):
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))


    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

class gramSaverMDP(gramListener):
    def __init__(self):
        pass
        

    def enterStatenoreward(self, ctx):
        self.states = [str(x) for x in ctx.ID()]
        

    def enterStatereward(self, ctx):
        self.states = [str(x) for x in ctx.ID()]
        self.rewards = [int(str(x)) for x in ctx.INT()]

    def enterDefactions(self, ctx):
        self.mdp = MDP(self.states, [str(x) for x in ctx.ID()], self.rewards)


    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        
        try:
            i_act = np.where(self.mdp.actions == act)[0][0]
        except:
            raise Exception(f"Action {act} not defined in header.")
        try:
            i_dep = np.where(self.mdp.states == dep)[0][0]
        except:
            raise Exception(f"Start state {dep} not defined in header.")

        if np.sum(self.mdp.P[i_act][i_dep]) > 0:
            raise Exception(f'Transitions from state {dep} with action {act} defined several times.')
        
        for i in range(len(ids)):
            target = ids[i]
            try:
                i_target = np.where(self.mdp.states == target)[0][0]
            except:
                raise Exception(f"Target state {target} not defined in header.")
            weight = weights[i]
            self.mdp.P[i_act, i_dep, i_target] = weight
        

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        
        try:
            i_dep = np.where(self.mdp.states == dep)[0][0]
        except:
            raise Exception(f"Start state {dep} not defined in header.")

        if np.sum(self.mdp.P[0][i_dep]) > 0:
            raise Exception(f'Transitions from state {dep} defined several times.')
        

        for i in range(len(ids)):
            target = ids[i]
            try:
                i_target = np.where(self.mdp.states == target)[0][0]
            except:
                raise Exception(f"Target state {target} not defined in header.")
            weight = weights[i]
            self.mdp.P[0, i_dep, i_target] = weight
    
    def get_mdp(self):
        self.mdp.validationAndNormalisation()
        return self.mdp

class MDP:
    epsilonAction = "__epsilonAction__" # Action quand il n'y a pas de non déterminisme

    def __init__(self, states, actions, rewards=None):
        self.states = np.array(states)
        self.actions = np.array([MDP.epsilonAction] + actions)
        self.P = np.zeros((len(self.actions), len(self.states), len(self.states)))
        if rewards == None: self.rewards = None
        else: self.rewards = np.array(rewards)
        self.accessible = self.possibleActions()
    
    def __repr__(self):
        string = "\nMarkovian Decision Process\nActions : " + str(self.actions) + "\States : " + str(self.states)
        for i in range(len(self.actions)):
            string += f"\nAction {self.actions[i]} : \n{self.P[i]}"
        return string

    def validationAndNormalisation(self):
        mdp_dep_act_tar = self.P.transpose((1,0,2))
        
        for i_dep in range(mdp_dep_act_tar.shape[0]):
            if np.sum(mdp_dep_act_tar[i_dep]) == 0:
                print(f'WARNING : no transition defined from state {self.states[i_dep]}')
                print(f'\t>> Add a loop from {self.states[i_dep]}')
                mdp_dep_act_tar[i_dep][0][i_dep] = 1

            for i_act in range(mdp_dep_act_tar.shape[1]):
                sum = np.sum(mdp_dep_act_tar[i_dep][i_act])
                if sum > 0:
                    mdp_dep_act_tar[i_dep][i_act] /= sum

            if np.sum(mdp_dep_act_tar[i_dep][1:]) > 0 and np.sum(mdp_dep_act_tar[i_dep][0]) > 0:
                raise Exception(f'Conflict of transitions from state {self.states[i_dep]} ')

    def possibleActions(self):
        accessible = [[]]*len(self.states)
        Pflat = np.sum(a=self.P, axis=2)
        for i_state in range(self.P.shape[1]):
            for i_act in range(self.P.shape[0]):
                if Pflat[i_act,i_state] > 0:
                    accessible[i_state].append(i_act)
        return accessible
        
        


class Simulation:
    def __init__(self, mdp, automatic=True, verbose=True):
        self.mdp = mdp
        self.automatic = automatic
        self.i_currentState = 0
        self.i_previousState = None
        self.actionUsed = None
        self.verbose = verbose

    def next(self):
        i_possibleAction = self.mdp.accessible[self.i_currentState]
        possibleAction = [self.mdp.actions[i_action] for i_action in i_possibleAction]
        if self.automatic:
            if i_possibleAction == [0]:
                i_action = 0
                if self.verbose: print('Automatic transition')
                self.actionUsed = None
            else:
                i_action = np.random.choice(size=1, a=i_possibleAction)[0]
                if self.verbose: print(f'Drawing of action {self.mdp.actions[i_action]}')
                self.actionUsed = self.mdp.actions[i_action]
        else:
            if i_possibleAction == [0]:
                i_action = 0
                print('Automatic transition')
                self.actionUsed = None
            else:
                print(f'Possibles actions : {[self.mdp.actions[pa] for pa in i_possibleAction]}')
                action = ""
                while action not in possibleAction:
                    print("Enter an action :")
                    action = input()

                i_action = np.where(self.mdp.actions == action)[0][0]
                self.actionUsed = self.mdp.actions[i_action]

        new_state = np.random.choice(a=self.mdp.P.shape[2], size=1, p=self.mdp.P[i_action][self.i_currentState])[0]
        if self.verbose: print(f" >> Transition from state {self.mdp.states[self.i_currentState]} to state {self.mdp.states[new_state]}\n")

        self.i_previousState = self.i_currentState
        self.i_currentState = new_state

    def monteCarlo(self, s, n, epsilon, delta):
        i_s = np.where(self.mdp.states == s)[0][0]
        N = int((np.log(2) - np.log(delta)) / (2*epsilon)**2)
        print(f'\r>> Nombre de simulations : {N}')
        r = 0

        for _ in range(N):
            self.__init__(self.mdp, True, False)

            for _ in range(n + 1):
                if self.i_currentState == i_s:
                    r += 1
                    break
                self.next()

        return r / N

    def SPRT(self, theta, epsilon, alpha, beta, s, n):
        i_s = np.where(self.mdp.states == s)[0][0]

        dm = 0
        m = 0
        gamma1 = theta - epsilon
        gamma0 = theta + epsilon

        logA = np.log((1 - beta) / alpha)
        logB = np.log(beta / (1 - alpha))


        logRm = dm * np.log(gamma1) + (m-dm)*np.log(1 - gamma1) - dm * np.log(gamma0) - (m-dm)*np.log(1 - gamma0)

        while logB < logRm and  logRm < logA:
            self.__init__(self.mdp, True, False)

            m += 1
            for _ in range(n):
                self.next()
                if self.i_currentState == i_s:
                    dm += 1
                    logRm += np.log(gamma1 / gamma0) - np.log(1 - gamma1) + np.log(1 - gamma0)
                    break
            
            logRm += np.log(1 - gamma1) - np.log(1 - gamma0)

        if logA <= logRm:
            print('Accept H1')

        if logRm <= logB:
            print('Accept H0')

        return m, dm/m


def main():
    lexer = gramLexer(FileStream("ex2.mdp")) 
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    saver = gramSaverMDP()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    walker.walk(saver, tree)
    mdp = saver.get_mdp()

    simu = Simulation(mdp, automatic=True)
    # print('\n\n#################   Simulation Start   #################\n')
    # for i in range(10000):
    #     simu.next()
    # print(simu.monteCarlo('S4', 5, 0.01, 0.01))
    # print(simu.SPRT(0.16, 1e-3, 0.01, 0.01, 'S7', 10))


if __name__ == '__main__':
    main()
