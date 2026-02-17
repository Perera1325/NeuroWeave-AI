import matplotlib.pyplot as plt
import numpy as np


def plot_loss(losses):

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def accuracy(y_true, y_pred):

    predictions = (y_pred > 0.5).astype(int)
    return np.mean(predictions == y_true)
