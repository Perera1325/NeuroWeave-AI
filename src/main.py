from brain_visualizer import BrainVisualizer
from neural_network import NeuralNetwork
from utils import plot_loss, accuracy
from data_loader import load_data


if __name__ == "__main__":

    # Load dataset
    X_train, X_test, y_train, y_test = load_data()

    # Create neural network
    nn = NeuralNetwork(input_size=X_train.shape[1])

    print("Training on real dataset...")

    losses = nn.train(X_train, y_train, epochs=2000)

    # Predictions
    predictions = nn.forward(X_test)

    acc = accuracy(y_test, predictions)

    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    # Plot loss
    plot_loss(losses)

    # Brain animation intensity based on accuracy
    brain = BrainVisualizer()
    brain.animate()
