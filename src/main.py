import os
import numpy as np

from brain_visualizer import BrainVisualizer
from neural_network import NeuralNetwork
from utils import (
    plot_loss,
    accuracy,
    predict_sample,
    save_model,
    load_model
)
from data_loader import load_data


MODEL_PATH = "models/neural_model.pkl"


def train_model():

    X_train, X_test, y_train, y_test = load_data()

    nn = NeuralNetwork(input_size=X_train.shape[1])

    print("Training model...")
    losses = nn.train(X_train, y_train, epochs=1500)

    predictions = nn.forward(X_test)
    acc = accuracy(y_test, predictions)

    print(f"\nFinal Accuracy: {acc * 100:.2f}%")

    plot_loss(losses)

    save_model(nn, MODEL_PATH)

    print("Model saved successfully.")

    return nn


def main():

    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        nn = load_model(MODEL_PATH)
    else:
        nn = train_model()

    brain = BrainVisualizer()

    while True:

        print("\nEnter 30 feature values separated by space (or type 'exit'):")

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

            brain.animate(intensity=conf)

        except Exception:
            print("Invalid input. Please enter exactly 30 numbers.")


if __name__ == "__main__":
    main()
