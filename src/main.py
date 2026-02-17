import numpy as np
from brain_visualizer import BrainVisualizer
from neural_network import NeuralNetwork
from utils import plot_loss


if __name__ == "__main__":

    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    nn = NeuralNetwork()

    print("Training Neural Network...")

    losses = nn.train(X, y, epochs=3000)

    print("\nFinal Predictions:")
    print(nn.forward(X))

    # Plot loss
    plot_loss(losses)

    # Animate brain
    brain = BrainVisualizer()
    brain.animate()
