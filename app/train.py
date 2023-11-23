import os
import sys
import torch
from torch.utils.data import DataLoader
from rich import print

from core import config
from core.model import DigitRecognitionModel
from core.prepare_data import get_data
from core.train_model import train


def main():
    # Model name
    if len(sys.argv) > 2:
        print("Usage: <model_name.pth>")
        return
    elif len(sys.argv) == 1:
        model_name = "model.pth"
        
    else: 
        model_name = sys.argv[1]
        if not model_name.endswith(".pth"):
            print("Usage: <model_name.pth>")
            return
    
    # Get data to fit the model
    train_data, test_data, n_classes = get_data()

    # Load data in batches
    train_dataloader = DataLoader(dataset = train_data,
                                batch_size = config.BATCH_SIZE,
                                shuffle = True)

    test_dataloader = DataLoader(dataset = test_data,
                                batch_size = config.BATCH_SIZE)
    
    # Instanciate the model
    model = DigitRecognitionModel(in_features = config.IMAGE_SIZE**2, out_features = n_classes)

    # Instanciate optimizer and loss function
    optimizer = config.OPTIMIZER(model.parameters(), lr = config.LEARNING_RATE)
    loss_fn = config.LOSS_FUNCTION()

    # Training process
    print("[INFO]: [green]Training model...[/green]\n")
    train(model = model, 
        train_dataloader = train_dataloader, 
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        loss_fn = loss_fn,
        epochs = config.EPOCHS,
        seed = config.SEED)
    
    if not os.path.isdir(config.MODELS_PATH):
        os.mkdir(config.MODELS_PATH)
        
    torch.save(model.state_dict(), f"{config.MODELS_PATH}/{model_name}")

if __name__ == "__main__":
    main()