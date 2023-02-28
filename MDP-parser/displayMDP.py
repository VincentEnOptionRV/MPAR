from mdp import *
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph, get_curved_edge_paths, get_fruchterman_reingold_layout

lexer = gramLexer(FileStream("ex2.mdp")) 
stream = CommonTokenStream(lexer)
parser = gramParser(stream)
tree = parser.program()
saver = gramSaverMDP()
walker = ParseTreeWalker()
walker.walk(saver, tree)
mdp0 = saver.get_mdp()

class Showtime:
    def __init__(self, mdp):
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
        self.node_color = {}
        self.node_size = {}
        self.node_edge_color = {}
        self.curved_edges = []
        self.straight_edges = []

        self.G = nx.DiGraph()
        self.r = 0.5
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
                    else:
                        action_etat = etat

                    for ind_destination in range(len(self.mdp.states)):
                        destination = self.mdp.states[ind_destination]
                        weight = transitions[ind_etat, ind_action, ind_destination]
                        if weight > 0:
                            self.G.add_edge(action_etat, destination)
                            self.edge_labels[(action_etat, destination)] = weight
                            if action_etat == etat:
                                self.edge_weights[(action_etat, destination)] = 1
                            else:
                                self.edge_weights[(action_etat, destination)] = 20
                            self.edge_width[(action_etat, destination)] = 0.4*self.r
                            self.curved_edges.append((action_etat, destination))
                            self.edge_alpha[(action_etat, destination)] = 1
            for i in range(len(actions_departing)-1):
                self.G.add_edge(actions_departing[i], actions_departing[i+1])
                self.edge_weights[(actions_departing[i], actions_departing[i+1])] = 40
                self.edge_width[(actions_departing[i], actions_departing[i+1])] = 0.001
                self.edge_alpha[(actions_departing[i], actions_departing[i+1])] = 0
                self.straight_edges.append((actions_departing[i], actions_departing[i+1]))
    
    def layouts(self):
        self.node_layout = get_fruchterman_reingold_layout(list(self.G.edges), edge_weights=self.edge_weights, origin=(0.2,0.2), scale=(0.5,0.5))
        self.edge_layout = get_curved_edge_paths(self.curved_edges, node_positions=self.node_layout, k=0.05, bundle_parallel_edges=False, selfloop_radius = 0.001)
        self.edge_layout.update(get_curved_edge_paths(self.straight_edges, node_positions=self.node_layout, k=0.001))

    def show(self):
        Graph(self.G, node_layout=self.node_layout, 
              edge_width=self.edge_width, edge_color="black", edge_alpha=self.edge_alpha, edge_layout=self.edge_layout, arrows=True,
              node_size=self.node_size, node_color = self.node_color, node_edge_color=self.node_edge_color, node_edge_width=0.3*self.r,
              node_labels=self.labels, node_label_fontdict=dict(size=12), edge_labels=self.edge_labels)

plt.figure(figsize=(10,10))

a = Showtime(mdp0)
plt.ion()
plt.show()

for _ in range(1):
    plt.clf()
    a.show()
    plt.pause(1)

plt.ioff()
plt.show()