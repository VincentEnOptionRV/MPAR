from mdp import *
import matplotlib.pyplot as plt
import networkx as nx

lexer = gramLexer(FileStream("ex2.mdp")) 
stream = CommonTokenStream(lexer)
parser = gramParser(stream)
tree = parser.program()
saver = gramSaverMDP()
walker = ParseTreeWalker()
walker.walk(saver, tree)
mdp = saver.get_mdp()

G = nx.DiGraph()
G.add_nodes_from(mdp.states)
node_sizes = [300 for i in range(len(mdp.states))]

nx.draw(G, node_size=node_sizes, with_labels=True, font_weight='bold')
plt.show()

# node_size_act = [10 for i in range(len(mdp.actions))]


# G.add_nodes_from(mdp.actions)
# nx.draw(G, node_size=node_sizes+node_size_act)



