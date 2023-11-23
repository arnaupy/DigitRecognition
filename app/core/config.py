from torch.nn import CrossEntropyLoss
from torch.optim import SGD

# Prepare data parameters
ROOT = "./"
DATA_DIRNAME = "MNIST" 
IMAGE_SIZE = 28

# Device and random seed
DEVICE = "cpu"
SEED = 42

# Model parameters
HIDDEN_UNITS = 10

# Training parameters
LEARNING_RATE = 0.1
BATCH_SIZE = 32
EPOCHS = 5

# Model optimizer
OPTIMIZER = SGD

# Model loss function
LOSS_FUNCTION = CrossEntropyLoss

# Save models path
MODELS_PATH = "models"