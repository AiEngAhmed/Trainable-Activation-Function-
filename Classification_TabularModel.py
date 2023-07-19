import torch.nn as nn
import torch

# Define the model architecture
class MLP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super(MLP_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out = self.classifier(x)
        return out

# Just we using TabNet decoder architecture step 1 without encoder
class TabNet_decoder_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, keep_prob=0.1):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int for task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        hidden_dim : int
            Dimension of the prediction  layer (usually between 64 and 256)
        activation : function
            Activation function in the network
        keep_prob : int
            Number of dropout from layer (default 0.1)
        """
        super(TabNet_decoder_Model, self).__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim) 
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim) 
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_out = nn.Linear(hidden_dim, output_dim) 
        
        self.activation = activation

        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm4 = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs):
        
        # Shared across decision steps 
        x1 = self.activation(self.layer_1(inputs))
        x1 = self.batchnorm1(x1)
        x2 = self.activation(self.layer_2(x1))
        x2 = self.batchnorm2(x2)

        # Decision step dependent
        x3 = self.activation(self.layer_3(x1 + x2))
        x3 = self.batchnorm3(x3)
        x4 = self.activation(self.layer_4(x2 + x3))
        x4 = self.batchnorm4(x4)

        # FC
        x = self.layer_out(x3 + x4)
        # Reconstructed_features => x
        
        return x


class BinaryClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, keep_prob=0.1):
        super(BinaryClassification, self).__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim) 
        self.layer_2 = nn.Linear(hidden_dim, 64)
        self.layer_out = nn.Linear(64, output_dim) 
        
        self.activation = activation
        self.dropout = nn.Dropout(p=keep_prob)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.activation(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.activation(self.layer_2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = self.layer_out(x)
        
        return x
