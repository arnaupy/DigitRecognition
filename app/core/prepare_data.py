import os
from rich import print
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from . import config 

def get_data():
    # Download data
    data_path = os.path.join(config.ROOT, config.DATA_DIRNAME)
    print("[INFO]: [green]Preparing data...[/green]")

    # Check if dataset was already downloaded
    is_downloaded = os.path.isdir(data_path)

    # Getting data
    train_data = MNIST("./", transform = ToTensor(), download = True)
    test_data = MNIST("./", transform = ToTensor(), download = True, train = False)
    n_classes = len(train_data.classes)

    # Check if dataset was already downloaded
    if is_downloaded:
        print(f"[INFO]: [green]Dataset[/green] [red]{config.DATA_DIRNAME}[/red] [green]is already downloaded![/green]\n")

    else:
        print(f"[INFO]: [green]Dataset[/green] [red]{config.DATA_DIRNAME}[/red] [green]downloaded![/green]\n")
    
    return train_data, test_data, n_classes




