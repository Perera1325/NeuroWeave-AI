import numpy as np


class NeuralNetwork:

    def __init__(self, input_size=2, hidden_size=5, output_size=1):

        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)

        return self.output

    def backward(self, X, y):

        # Output layer error
        error = y - self.output
        d_output = error * self.sigmoid_derivative(self.output)

        # Hidden layer error
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Update weights
        self.W2 += self.a1.T.dot(d_output) * self.learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.W1 += X.T.dot(d_hidden) * self.learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

        loss = np.mean(np.square(error))
        return loss

    def train(self, X, y, epochs=5000):

        losses = []

        for epoch in range(epochs):

            self.forward(X)
            loss = self.backward(X, y)
            losses.append(loss)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return losses
