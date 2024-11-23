import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, input, hidden_sizes, output, df , coefficients, biases, lr):
        super(FeedForwardNN, self).__init__()

        self.input = input
        self.output = output
        self.hidden_sizes = hidden_sizes
        self.df = df
        self.coefficients = coefficients
        self.biases = biases
        self.lr = lr

        # list of hidden layers
        layers = []
        layers.append(nn.Linear(len(input), hidden_sizes[0]))    # <-- input layer

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        layers.append(nn.Linear(hidden_sizes[-1], len(output)))     # <-- output layer
        
        self.layers = nn.ModuleList(layers)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.leaky_relu(layer(x))
            
        x = self.layers[-1](x)
        
        return x