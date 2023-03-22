from mdp import *
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph, get_curved_edge_paths, get_fruchterman_reingold_layout
import sys
from qlearning import qlearning
import modelChecking

class MDPGraph:
    def __init__(self, fichier, alreadyMDP=None):
        if alreadyMDP is None:
            lexer = gramLexer(FileStream(fichier)) 
            stream = CommonTokenStream(lexer)
            parser = gramParser(stream)
            tree = parser.program()
            saver = gramSaverMDP()
            walker = ParseTreeWalker()
            walker.walk(saver, tree)
            mdp = saver.get_mdp()
        else:
            mdp = alreadyMDP
        self.mdp = mdp
        self.mdp_graph()
        self.label()
        self.layouts()

    def label(self):
        self.labels = {}
        for node in self.G.nodes():
            if node in self.mdp.states:
                self.labels[node] = node
            elif str(node)[:-1] == self.mdp.actions[0]:
                self.labels[node] = ""
            else:
                self.labels[node] = str(node)[:-1]

    def mdp_graph(self):
        self.edge_labels = {}
        self.edge_weights = {}
        self.edge_width = {}
        self.edge_alpha = {}
        self.edge_color = {}
        self.node_color = {}
        self.node_size = {}
        self.node_edge_color = {}
        self.curved_edges = []
        self.straight_edges = []

        self.G = nx.DiGraph()
        self.r = 0.7
        transitions = np.swapaxes(self.mdp.P, 0, 1)

        for ind_etat in range(len(self.mdp.states)):
            etat = self.mdp.states[ind_etat]
            self.G.add_node(etat)
            self.node_color[etat] = "white"
            self.node_edge_color[etat] = "black"
            self.node_size[etat] = 3*self.r
            actions_departing = []
            for ind_action in range(len(self.mdp.actions)):
                action = self.mdp.actions[ind_action]
                if np.sum(transitions[ind_etat, ind_action]) > 0:
                    if ind_action > 0:
                        action_etat = action + str(ind_etat)
                        self.G.add_node(action_etat)
                        actions_departing.append(action_etat)
                        self.node_color[action_etat] = "white"
                        self.node_edge_color[action_etat] = "white"
                        self.node_size[action_etat] = 1.5*self.r
                        self.G.add_edge(etat, action_etat)
                        self.edge_width[(etat, action_etat)] = 0.6*self.r
                        self.edge_weights[(etat, action_etat)] = 90
                        self.straight_edges.append((etat, action_etat))
                        self.edge_alpha[(etat, action_etat)] = 1
                        self.edge_color[(etat, action_etat)] = "black"
                    else:
                        action_etat = etat

                    for ind_destination in range(len(self.mdp.states)):
                        destination = self.mdp.states[ind_destination]
                        weight = transitions[ind_etat, ind_action, ind_destination]
                        if weight > 0:
                            self.G.add_edge(action_etat, destination)
                            self.edge_labels[(action_etat, destination)] = round(weight,2)
                            self.edge_width[(action_etat, destination)] = 0.4*self.r
                            self.curved_edges.append((action_etat, destination))
                            self.edge_alpha[(action_etat, destination)] = 1
                            self.edge_color[(action_etat, destination)] = "black"
                            if action_etat == etat:
                                self.edge_weights[(action_etat, destination)] = 1
                            else:
                                self.edge_weights[(action_etat, destination)] = 20

            for i in range(len(actions_departing)-1):
                self.G.add_edge(actions_departing[i], actions_departing[i+1])
                self.edge_weights[(actions_departing[i], actions_departing[i+1])] = 40
                self.edge_width[(actions_departing[i], actions_departing[i+1])] = 0.001
                self.edge_alpha[(actions_departing[i], actions_departing[i+1])] = 0
                self.edge_color[(actions_departing[i], actions_departing[i+1])] = "black"
                self.straight_edges.append((actions_departing[i], actions_departing[i+1]))
        
        self.initial_node_edge_color = self.node_edge_color.copy()
    
    def layouts(self):
        self.node_layout = get_fruchterman_reingold_layout(list(self.G.edges), edge_weights=self.edge_weights, origin=(0.3,0.3), scale=(1,0.75))
        self.edge_layout = get_curved_edge_paths(self.curved_edges, node_positions=self.node_layout, k=0.05, bundle_parallel_edges=False, selfloop_radius = 0.001, scale=(2,1.5))
        self.edge_layout.update(get_curved_edge_paths(self.straight_edges, node_positions=self.node_layout, k=0.001))

    def show(self):
        Graph(self.G, node_layout=self.node_layout, 
              edge_width=self.edge_width, edge_color=self.edge_color, edge_alpha=self.edge_alpha, edge_layout=self.edge_layout, arrows=True,
              node_size=self.node_size, node_color = self.node_color, node_edge_color=self.node_edge_color, node_edge_width=0.3*self.r,
              node_labels=self.labels, node_label_fontdict=dict(size=20/np.log(len(self.G.nodes)+1)), edge_labels=self.edge_labels, edge_label_fontdict=dict(size=25/np.log(1+len(self.G.edges))))

    def update(self):
        for node in self.node_color.keys():
            self.node_color[node] = "white"
            self.node_edge_color[node] = self.initial_node_edge_color[node]
        
        for edge in self.edge_color.keys():
            self.edge_color[edge] = "black"

        self.node_color[self.mdp.states[self.simulation.i_currentState]] = "#00b31b"
        if self.simulation.i_currentState != self.simulation.i_previousState and self.simulation.i_previousState is not None:
            self.node_color[self.mdp.states[self.simulation.i_previousState]] = "#bdffd8"
        
        if self.simulation.actionUsed is None and self.simulation.i_previousState is not None:
            self.edge_color[(self.mdp.states[self.simulation.i_previousState], self.mdp.states[self.simulation.i_currentState])] = "#366bff"

        if self.simulation.actionUsed is not None:
            action_etat = self.simulation.actionUsed + str(self.simulation.i_previousState)
            self.node_edge_color[action_etat] = "#366bff"
            self.edge_color[(self.mdp.states[self.simulation.i_previousState], action_etat)] = "#366bff"
            self.edge_color[(action_etat, self.mdp.states[self.simulation.i_currentState])] = "#366bff"

    def update2(self):
        for node in self.node_color.keys():
            self.node_color[node] = "white"
            self.node_edge_color[node] = self.initial_node_edge_color[node]
        
        for edge in self.edge_color.keys():
            self.edge_color[edge] = "black"

        self.node_color[self.mdp.states[self.simulation.i_currentState]] = "#00b31b"
        

def print_adv(mdp, adv, advVal, infboucle):
    oneNotZero = False
    for i in range(len(adv)):
        choix = adv[i]
        if choix > 0:
            oneNotZero = True
            print(f"From state {mdp.states[i]}, chose action {mdp.actions[choix]}. Score = {advVal[i]:.2f}")
        elif i in infboucle and mdp.accessible[i][0] != 0:
            oneNotZero = True
            print(f"From state {mdp.states[i]}, chose action {mdp.actions[mdp.accessible[i][0]]}. Score = {advVal[i]:.2f}")
    if not oneNotZero:
        print("There are no decisions to take with this starting set of states.")

def check_file():
    if len(sys.argv) <= 1:
        raise Exception("Veuillez sélectionner un fichier .mdp à ouvrir")
    else:
        f = str(sys.argv[1])
    return f

def check_mode():
    if len(sys.argv) <= 2:
        raise Exception("Veuillez sélectionner un mode de fonctionnement. Modes possibles :\nsimu (simulation), \nmc (Model Checking accessibilité), \nsmc (Model Checking Statistique), \nmax (Model Checking Max), \nqlearn (Qlearning)")
    else:
        mode_dict = {
            "simu":0,
            "mc":1,
            "smc":2,
            "max":3,
            "qlearn":4
        }
        try:
            mode = mode_dict[str(sys.argv[2])]
        except:
            raise Exception("Erreur dans le mode. Modes possibles :\nsimu (simulation), \nmc (Model Checking accessibilité), \nsmc (Model Checking Statistique), \nmax (Model Checking Max), \nqlearn (Qlearning)")
    return mode

def main():
    # python mdpGraph.py fichier mode
    file = check_file()
    graphe = MDPGraph(file)

    mode = check_mode()
    adversary = None

    plt.ion()
    graphe.show()
    plt.show()

    end = False
    while not end:
        end = True
        if mode == 0:
            print("Entrer le temps de pause (en seconde) entre deux frames de la simulation :")
            half_pause_time = int(input())/2
            print("Entrer le nombre d'itérations de la simulation :")
            n = int(input())
            ok = False
            print("Mode automatique ? O/N")
            while not ok:
                auto = str(input())
                ok = auto in ['O','o','N','n']
                auto = auto == 'O' or auto == 'o'
            
            graphe.simulation = Simulation(graphe.mdp, auto)

            print('\n#################   Simulation Start   #################\n')
            
            for _ in range(n):

                graphe.update()
                plt.clf()
                graphe.show()
                plt.pause(half_pause_time)

                graphe.update2()
                plt.clf()
                graphe.show()
                plt.pause(half_pause_time)

                graphe.simulation.next()

            plt.ioff()
            print('#################   Simulation End   #################\n')
            print("Close window to exit.")
            plt.show()
        
        elif mode == 1:
            print("#################   Model Checking   #################")
            if adversary is None:
                adversary = modelChecking.buildAdversary(graphe.mdp)
            
            print("Définition des états cibles :")
            etatsCibles = []
            while len(etatsCibles) == 0:
                print("Liste des états cibles vide.")
                notDoneYet = True
                while notDoneYet:
                    print("Ajouter un état dans la liste des états cibles (laisser vide pour terminer)")
                    e = input()
                    if e == '':
                        notDoneYet = False
                    elif e in graphe.mdp.states:
                        etatsCibles.append(e)
                    else:
                        print("État introuvable.")
            
            print("Imposer une limite du nombre de transitions ? Si oui, entrer le nombre maximal de transitions autorisé. Sinon, laisser vide.")
            ok = False
            while not ok:
                ok = True
                n = input()
                if n!='':
                    try:
                        n = int(n)
                    except:
                        ok = False
                else:
                    n = None

            modelChecking.modelChecking(graphe.mdp, adversary, etatsCibles, n)
            plt.close()

        elif mode == 2:
            print("#################   Model Checking Statistique   #################")
            print("Choisir une méthode de model checking statistique : Monte-Carlo ou SPRT (entrer 1 ou 2)")
            ok = False
            while not ok:
                ok = True
                m = input()
                try:
                    m = int(m)
                    assert (m == 1 or m == 2)
                except:
                    ok = False
            print("Définition des états cibles :")
            etatsCibles = []
            while len(etatsCibles) == 0:
                print("Liste des états cibles vide.")
                notDoneYet = True
                while notDoneYet:
                    print("Ajouter un état dans la liste des états cibles (laisser vide pour terminer)")
                    e = input()
                    if e == '':
                        notDoneYet = False
                    elif e in graphe.mdp.states:
                        etatsCibles.append(e)
                    else:
                        print("État introuvable.")
            graphe.simulation = Simulation(graphe.mdp, True)
            print("Entrer le nombre maximal de transitions autorisé.")
            ok = False
            while not ok:
                ok = True
                n = input()
                try:
                    n = int(n)
                except:
                    ok = False
            if m == 1:
                print("Entrer la valeur de epsilon.")
                ok = False
                while not ok:
                    ok = True
                    eps = input()
                    try:
                        eps = float(eps)
                    except:
                        ok = False
                print("Entrer la valeur de delta.")
                ok = False
                while not ok:
                    ok = True
                    d = input()
                    try:
                        d = float(d)
                    except:
                        ok = False
                proba = graphe.simulation.monteCarlo(etatsCibles, n, eps, d)
                print(f"Probabilité d'atteindre le(s) sommet(s) d'arrivée depuis le sommet de départ : {proba:.2f}")
            else:
                print("Entrer la valeur de theta.")
                ok = False
                while not ok:
                    ok = True
                    t = input()
                    try:
                        t = float(t)
                    except:
                        ok = False
                print("Entrer la valeur de epsilon.")
                ok = False
                while not ok:
                    ok = True
                    eps = input()
                    try:
                        eps = float(eps)
                    except:
                        ok = False

                print("Entrer la valeur de alpha.")
                ok = False
                while not ok:
                    ok = True
                    a = input()
                    try:
                        a = float(a)
                    except:
                        ok = False

                print("Entrer la valeur de beta.")
                ok = False
                while not ok:
                    ok = True
                    b = input()
                    try:
                        b = float(b)
                    except:
                        ok = False
                m, dm, accept = graphe.simulation.SPRT(t, eps, a, b, etatsCibles, n)
                print(f"")

            plt.close()

        elif mode == 3:
            print("#################   Model Checking Max   #################")

            plt.close()

        elif mode == 4:
            if graphe.mdp.rewards is None:
                print("Impossible de faire du Qlearning sur un MDP sans récompenses.")
            print("#################   Qlearning   #################")
            print("Entrer la valeur de gamma pour l'algorithme")
            ok = False
            while not ok:
                ok = True
                gamma = input()
                try:
                    gamma = float(gamma)
                except:
                    ok = False
            print("Entrer le nombre d'itération de l'algorithme")
            ok = False
            while not ok:
                ok = True
                n = input()
                try:
                    n = int(n)
                except:
                    ok = False
            print("Calcul du meilleur adversaire...")
            adv, advVal, Q, infboucle = qlearning(graphe.mdp, gamma, n)
            print("Exécution de l'algorithme terminée. Vérification de la validité du résultat...")

            valid = False
            while not valid:
                valid = True
                failed = []
                for i in range(len(adv)):
                    choix = adv[i]
                    i_possibleAction = graphe.mdp.accessible[i]
                    if choix == 0 and i_possibleAction != [0]:
                        if i not in infboucle:
                            if valid:
                                print(f"Attention, le nombre d'itérations choisi ({n}) était trop faible. \nL'algorithme de qlearning va être relancé pour {2*n} itérations à partir de l'étape actuelle")
                            valid = False
                            failed.append(graphe.mdp.states[i])
                        else:
                            adv[i] = graphe.mdp.accessible[i][0]
                if not valid:
                    print(f"Sommets pour lesquels on ne sait pas encore choisir d'action : {failed}\n")
                    n*=2
                    adv, advVal, Q, infboucle = qlearning(graphe.mdp, gamma, n, Q)
            print_adv(graphe.mdp, adv, advVal, infboucle)
            print("Souhaitez vous générer la MC induite par cet adversaire et lancer un autre mode à partir de cette dernière ? O/N")
            ok = False
            while not ok:
                v = str(input())
                ok = v in ['O','o','N','n']
                v = v == 'O' or v == 'o'
            if v:
                end = False
                print("Quel mode souhaitez vous lancer ? Modes possibles : simu (simulation), mc (Model Checking), smc (Model Checking Statistique)")
                mode_dict = {
                    "simu":0,
                    "mc":1,
                    "smc":2
                }
                wrong = True
                while wrong:
                    wrong = False
                    m = input()
                    try:
                        mode = mode_dict[m]
                    except:
                        wrong = True
                
                if mode == 0 or mode == 2: # alors on doit générer la mc, sinon pas la peine ça se fait tout seul dans le checking d'accessibilité.
                    mc = np.array([[graphe.mdp.P[adv[i_state]][i_state] for i_state in range(len(graphe.mdp.states))]])
                    graphe.mdp = MDP(graphe.mdp.states, [])
                    graphe.mdp.P = mc
                    graphe.mdp.validationAndNormalisation()
                    graphe.mdp.possibleActions()
                    plt.clf()
                    graphe = MDPGraph("",graphe.mdp)
                    graphe.show()
                else:
                    adversary = adv
            else:
                plt.ioff()
                print("Close window to exit.")
                plt.show()



if __name__ == '__main__':
    main()