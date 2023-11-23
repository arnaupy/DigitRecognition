from torch import nn

from . import config 

class DigitRecognitionModel(nn.Module):
    """Model to classify number digits form 0 to 9"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Flatter matrix data into a 1D vector
        self.flatten_layer = nn.Flatten()
        
        # First block of the model
        self.block_1 = nn.Linear(in_features = in_features, out_features = config.HIDDEN_UNITS)

        
        # Output layer of the model
        self.output_layer = nn.Linear(in_features = config.HIDDEN_UNITS, out_features = out_features)

    def forward(self, X):
        return self.output_layer(self.block_1(self.flatten_layer(X)))
        