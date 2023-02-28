from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import numpy as np

        
class gramPrintListener(gramListener):

    def __init__(self):
        pass
        
    def enterDefstates(self, ctx):
        print("States: %s" % str([str(x) for x in ctx.ID()]))

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
        
    def enterDefstates(self, ctx):
        self.states = [str(x) for x in ctx.ID()]

    def enterDefactions(self, ctx):
        self.mdp = MDP(self.states, [str(x) for x in ctx.ID()])


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

    def __init__(self, states, actions):
        self.states = np.array(states)
        self.actions = np.array([MDP.epsilonAction] + actions)
        self.P = np.zeros((len(self.actions), len(self.states), len(self.states)))
    
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

    def possibleActions(self, i_dep):
        act_tar = self.P.transpose((1,0,2))[i_dep]
        return [i_act for i_act in range(act_tar.shape[0]) if np.sum(act_tar[i_act]) > 0]
        
        


class Simulation:
    def __init__(self, mdp, automatic=True):
        self.mdp = mdp
        self.automatic = automatic
        self.i_currentState = 0
        self.i_previousState = None
        self.actionUsed = None

    def next(self):
        i_possibleAction = self.mdp.possibleActions(self.i_currentState)
        possibleAction = [self.mdp.actions[i_action] for i_action in i_possibleAction]
        if self.automatic:
            if i_possibleAction == [0]:
                i_action = 0
                print('Automatic transition')
                self.actionUsed = None
            else:
                i_action = np.random.choice(size=1, a=i_possibleAction)[0]
                print(f'Drawing of action {self.mdp.actions[i_action]}')
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
        print(f" >> Transition from state {self.mdp.states[self.i_currentState]} to state {self.mdp.states[new_state]}\n")

        self.i_previousState = self.i_currentState
        self.i_currentState = new_state



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

    print('\n\n#################   Simulation Start   #################\n')
    simu = Simulation(mdp, automatic=True)
    for i in range(10000):
        simu.next()

if __name__ == '__main__':
    main()
