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

def predict_sample(nn, sample):

    output = nn.forward(sample)

    confidence = float(output[0][0])

    prediction = 1 if confidence > 0.5 else 0

    return prediction, confidence

