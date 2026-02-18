import tkinter as tk
from tkinter import messagebox
import numpy as np
import os

from neural_network import NeuralNetwork
from utils import predict_sample, load_model
from data_loader import load_data
from brain_3d import Brain3D


MODEL_PATH = "models/neural_model.pkl"


class NeuroWeaveGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("NeuroWeave AI")
        self.root.geometry("600x400")

        self.model = self.load_or_train()

        self.create_widgets()

    def load_or_train(self):

        if os.path.exists(MODEL_PATH):
            return load_model(MODEL_PATH)

        # Train if not exists
        X_train, X_test, y_train, y_test = load_data()
        nn = NeuralNetwork(input_size=X_train.shape[1])
        nn.train(X_train, y_train, epochs=1000)

        return nn

    def create_widgets(self):

        title = tk.Label(
            self.root,
            text="NeuroWeave AI Predictor",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        instruction = tk.Label(
            self.root,
            text="Enter 30 values separated by space:"
        )
        instruction.pack()

        self.input_box = tk.Text(self.root, height=5, width=60)
        self.input_box.pack(pady=10)

        predict_btn = tk.Button(
            self.root,
            text="Predict",
            command=self.predict,
            bg="#4CAF50",
            fg="white",
            width=15
        )
        predict_btn.pack(pady=5)

        self.result_label = tk.Label(
            self.root,
            text="Result will appear here",
            font=("Arial", 12)
        )
        self.result_label.pack(pady=20)

    def predict(self):

        try:
            text = self.input_box.get("1.0", tk.END).strip()

            values = np.array(
                [float(x) for x in text.split()]
            ).reshape(1, -1)

            pred, conf = predict_sample(self.model, values)

            label = "Malignant" if pred == 1 else "Benign"

            result_text = f"Prediction: {label} | Confidence: {conf:.2f}"

            self.result_label.config(text=result_text)

            # Show 3D brain
            brain = Brain3D()
            brain.animate(intensity=conf)

        except Exception:
            messagebox.showerror(
                "Error",
                "Please enter exactly 30 numbers separated by space."
            )


if __name__ == "__main__":

    root = tk.Tk()
    app = NeuroWeaveGUI(root)
    root.mainloop()
