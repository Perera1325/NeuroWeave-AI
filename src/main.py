import numpy as np
from brain_visualizer import BrainVisualizer
from neural_network import NeuralNetwork
from utils import plot_loss, accuracy, predict_sample
from data_loader import load_data


if __name__ == "__main__":

    # Load dataset
    X_train, X_test, y_train, y_test = load_data()

    nn = NeuralNetwork(input_size=X_train.shape[1])

    print("Training model...")
    losses = nn.train(X_train, y_train, epochs=1500)

    predictions = nn.forward(X_test)
    acc = accuracy(y_test, predictions)

    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    plot_loss(losses)

    brain = BrainVisualizer()

    # Interactive loop
    while True:

        print("\nEnter values separated by space (or type 'exit'):")

        user_input = input("> ")

        if user_input.lower() == "exit":
            break

        try:
            values = np.array(
                [float(x) for x in user_input.split()]
            ).reshape(1, -1)

            pred, conf = predict_sample(nn, values)

            label = "Malignant" if pred == 1 else "Benign"

            print(f"Prediction: {label}")
            print(f"Confidence: {conf:.2f}")

            # Brain reacts to confidence
            brain.animate(intensity=conf)

        except Exception as e:
            print("Invalid input. Please try again.")
