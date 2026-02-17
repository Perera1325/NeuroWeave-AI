import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from matplotlib.animation import FuncAnimation


class BrainVisualizer:

    def __init__(self, neurons=120, connections=250):

        self.neurons = neurons
        self.connections = connections
        self.G, self.pos = self.generate_brain()

    def generate_brain(self):

        G = nx.Graph()
        positions = {}

        for i in range(self.neurons):
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 0.6)

            if x**2 + (y * 1.5) ** 2 < 2.5:
                positions[i] = (x, y)
                G.add_node(i)

        nodes = list(G.nodes())

        for _ in range(self.connections):
            n1, n2 = np.random.choice(nodes, 2)
            G.add_edge(n1, n2)

        return G, positions

    def random_signal(self):

        return random.sample(list(self.G.nodes()), 15)

    def animate(self, intensity=1.0):

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("black")

        def update(frame):

            ax.clear()
            ax.set_facecolor("black")

            active_nodes = self.random_s
