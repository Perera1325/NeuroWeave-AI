import numpy as np
from brain_visualizer import BrainVisualizer
from neural_network import NeuralNetwork


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

    # Create neural network
    nn = NeuralNetwork()

    print("Training Neural Network...")

    losses = nn.train(X, y, epochs=5000)

    print("\nFinal Predictions:")
    print(nn.forward(X))

    # Brain visualization
    brain = BrainVisualizer()

    # Activate neurons based on learning intensity
    active_nodes = brain.random_signal()

    brain.draw(active_nodes=active_nodes)
