import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


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

    def draw(self, active_nodes=None):

        G, pos = self.generate_brain()

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_facecolor("black")

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="cyan",
            alpha=0.1,
            width=1
        )

        # Default neurons
        node_colors = []
        for node in G.nodes():
            if active_nodes and node in active_nodes:
                node_colors.append("orange")
            else:
                node_colors.append("deepskyblue")

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=30,
            alpha=0.9
        )

        plt.axis("off")
        plt.title("NeuroWeave Circuit Brain", color="white")
        plt.show()

    def random_signal(self):

        # Random active neurons
        return random.sample(range(self.neurons), 10)
