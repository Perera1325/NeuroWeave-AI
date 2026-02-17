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

            if x**2 + (y*1.5)**2 < 2.5:
                positions[i] = (x, y)
                G.add_node(i)

        nodes = list(G.nodes())

        for _ in range(self.connections):
            n1, n2 = np.random.choice(nodes, 2)
            G.add_edge(n1, n2)

        return G, positions

    def random_signal(self):

        return random.sample(list(self.G.nodes()), 15)

    def animate(self):

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("black")

        def update(frame):

            ax.clear()
            ax.set_facecolor("black")

            active_nodes = self.random_signal()

            # Edges
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edge_color="cyan",
                alpha=0.08,
                width=1,
                ax=ax
            )

            # Node colors
            node_colors = []

            for node in self.G.nodes():
                if node in active_nodes:
                    node_colors.append("orange")
                else:
                    node_colors.append("deepskyblue")

            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                node_color=node_colors,
                node_size=40,
                alpha=0.9,
                ax=ax
            )

            ax.set_title("NeuroWeave Learning Brain", color="white")
            ax.axis("off")

        ani = FuncAnimation(fig, update, frames=30, interval=500)
        plt.show()
