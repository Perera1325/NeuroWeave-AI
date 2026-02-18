import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


class Brain3D:

    def __init__(self, neurons=150):

        self.neurons = neurons
        self.positions = self.generate_positions()

    def generate_positions(self):

        pos = []

        for _ in range(self.neurons):

            x = np.random.normal(0, 1)
            y = np.random.normal(0, 0.8)
            z = np.random.normal(0, 0.6)

            if x**2 + (y * 1.3) ** 2 + (z * 1.1) ** 2 < 3:
                pos.append([x, y, z])

        return np.array(pos)

    def random_active(self, count=20):

        idx = np.arange(len(self.positions))
        return random.sample(list(idx), count)

    def animate(self, intensity=1.0):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):

            ax.clear()

            active = self.random_active()

            xs = self.positions[:, 0]
            ys = self.positions[:, 1]
            zs = self.positions[:, 2]

            colors = []

            for i in range(len(xs)):
                if i in active:
                    colors.append("orange")
                else:
                    colors.append("cyan")

            size = 20 + 80 * intensity

            ax.scatter(xs, ys, zs, c=colors, s=size, alpha=0.9)

            ax.set_facecolor("black")
            ax.set_title("NeuroWeave 3D Brain", color="white")

            ax.view_init(elev=20, azim=frame * 3)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        ani = FuncAnimation(fig, update, frames=120, interval=100)

        plt.show()
