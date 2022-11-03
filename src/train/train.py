from src.models.model import ResNet18
from src.data.dataset import RainDetectionDataset, ImageRainDetectionDataset
from src.train.train_utils import train_model, initialize_model, set_parameter_requires_grad
from src.data.transforms import data_transforms

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
import torch.optim as optim

def train():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Number of classes in the dataset
    num_classes = 6
    # Batch size for training (change depending on how much memory you have)
    batch_size = 64
    # Number of epochs to train for
    num_epochs = 5
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Initialize the model for this run
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    #print(model_ft)



    # DATA
    print("Initializing Datasets and Dataloaders...")


    rain_dataset = ImageRainDetectionDataset(path='image_rain_dataset',
                                             transform=data_transforms)

    # rain_dataset = RainDetectionDataset('dataset',transform=data_transforms)
    train_size = int(len(rain_dataset)*0.7)
    val_size = len(rain_dataset) - train_size

    train_data, val_data = random_split(rain_dataset, [train_size, val_size])
    print("The length of train data is:",len(train_data))
    print("The length of val data is:",len(val_data))

    # Create training and validation datasets
    image_datasets = {'train': train_data,
                    'val':val_data}

    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)


    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()


    # Train and evaluate
    model_ft, hist = train_model(model_ft,
                                dataloaders_dict,
                                criterion,
                                optimizer_ft,
                                num_epochs=num_epochs,device=device)
    torch.save(model_ft.state_dict(), "data/model.pth")
    print("Saved PyTorch Model State to model.pth")
        
if __name__=='__main__':
    train()
