import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class BrainVisualizer:

    def __init__(self, neurons=120, connections=250):
        self.neurons = neurons
        self.connections = connections

    def generate_brain(self):
        G = nx.Graph()

        positions = {}
        for i in range(self.neurons):
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 0.6)

            if x**2 + (y*1.5)**2 < 2.5:
                positions[i] = (x, y)
                G.add_node(i)

        nodes = list(G.nodes())

        for _ in range(self.connections):
            n1, n2 = np.random.choice(nodes, 2)
            G.add_edge(n1, n2)

        return G, positions

    def draw(self):
        G, pos = self.generate_brain()

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_facecolor("black")

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="cyan",
            alpha=0.15,
            width=1
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="deepskyblue",
            node_size=20,
            alpha=0.9
        )

        plt.axis("off")
        plt.title("NeuroWeave Circuit Brain", color="white")
        plt.show()
