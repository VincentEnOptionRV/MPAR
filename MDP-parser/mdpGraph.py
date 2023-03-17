from mdp import *
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph, get_curved_edge_paths, get_fruchterman_reingold_layout
import sys

class MDPGraph:
    def __init__(self, fichier, automatic=True):
        lexer = gramLexer(FileStream(fichier)) 
        stream = CommonTokenStream(lexer)
        parser = gramParser(stream)
        tree = parser.program()
        saver = gramSaverMDP()
        walker = ParseTreeWalker()
        walker.walk(saver, tree)
        mdp = saver.get_mdp()

        self.mdp = mdp
        self.mdp_graph()
        self.label()
        self.layouts()

        self.simulation = Simulation(self.mdp, automatic)

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
        


def check_auto():
    if len(sys.argv) <= 4:
        print("Automatic progress engaged")
        auto = True
    elif sys.argv[4] == 'm':
        print("Manual progress engaged")
        auto = False
    else:
        auto = True
    return auto

def check_n():
    if len(sys.argv) <= 2:
        n = 100
    else:
        n = int(sys.argv[2])
    print(f"Number of iterations of the simulation : {n}")
    return n

def check_pause():
    if len(sys.argv) <= 3:
        t = 2.0
    else:
        t = float(sys.argv[3])
    print(f"Time per iteration : {t} s")
    return t

def check_file():
    if len(sys.argv) <= 1:
        f = "ex2.mdp"
    else:
        f = str(sys.argv[1])
    return f

def main():
    half_pause_time = check_pause()/2
    # plt.figure(figsize=(6,6))

    n = check_n()
    graphe = MDPGraph(check_file(), check_auto())
    plt.ion()
    plt.show()

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

if __name__ == '__main__':
    main()