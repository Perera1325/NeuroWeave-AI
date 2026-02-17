import numpy as np
from brain_visualizer import BrainVisualizer
from neural_network import NeuralNetwork


if __name__ == "__main__":

    # Create neural network
    nn = NeuralNetwork()

    # Sample input
    X = np.array([[0.5, 0.2, 0.8]])

    output = nn.forward(X)

    print("Neural Network Output:")
    print(output)

    # Visualize brain with signal
    brain = BrainVisualizer()
    active = brain.random_signal()
    brain.draw(active_nodes=active)
