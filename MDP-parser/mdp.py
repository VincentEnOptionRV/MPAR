from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import numpy as np
import sys

        
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
            raise Exception(f"Action {act} non définie dans l'en-tête.")
        try:
            i_dep = np.where(self.mdp.states == dep)[0][0]
        except:
            raise Exception(f"Etat de départ {dep} non défini dans l'en-tête.")
        
        for i in range(len(ids)):
            target = ids[i]
            try:
                i_target = np.where(self.mdp.states == target)[0][0]
            except:
                raise Exception(f"Etat cible {target} non défini dans l'en-tête.")
            weight = weights[i]
            self.mdp.P[i_act, i_dep, i_target] = weight
        

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        
        try:
            i_dep = np.where(self.mdp.states == dep)[0][0]
        except:
            raise Exception(f"Etat de départ {dep} non défini dans l'en-tête.")

        for i in range(len(ids)):
            target = ids[i]
            try:
                i_target = np.where(self.mdp.states == target)[0][0]
            except:
                raise Exception(f"Etat cible {target} non défini dans l'en-tête.")
            weight = weights[i]
            self.mdp.P[0, i_dep, i_target] = weight
    
    def get_mdp(self):
        self.mdp.validation()
        return self.mdp

class MDP:
    epsilonAction = "__epsilonAction__" # Action quand il n'y a pas de non déterminisme

    def __init__(self, states, actions):
        self.states = np.array(states)
        self.actions = np.array([MDP.epsilonAction] + actions)
        self.P = np.zeros((len(self.actions), len(self.states), len(self.states)))
    
    def __repr__(self):
        string = "\nMarkovian Decision Process\nActions : " + str(self.actions) + "\nEtats : " + str(self.states)
        for i in range(len(self.actions)):
            string += f"\nAction {self.actions[i]} : \n{self.P[i]}"
        return string

    def validation(self):
        pass

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
    print(mdp)

if __name__ == '__main__':
    main()
