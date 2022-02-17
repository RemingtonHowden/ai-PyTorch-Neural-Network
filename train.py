# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from torchvision import models, datasets, transforms
import PIL
from PIL import Image
import collections
from collections import OrderedDict
import seaborn as sns
import argparse 

# creating function to load data
def gather_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'



    # Defining transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)

    # Defining dataloader for training 
    trainloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

    # Defining transforms for Testing Set
    testing_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [ 0.229, 0.224, 0.225]),
    ])

    # Loading dataset with ImageFolder 
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # Defining dataloader for testing 
    testing_loader = DataLoader(testing_dataset, batch_size=32, shuffle=False)

    # Defining transforms for Validation Set
    validation_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    ])

    # Loading validation dataset with ImageFolder
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # Defining dataloader for validation set 
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    
    return trainloader, testing_loader, validation_loader, training_dataset
   
# File for mapping values to class labels    
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
   
# Providing 2 different model architectures for user selection    
arch={'densenet121' : 1024, 'vgg16' : 25088}

# Defines neural network with specified model and learning parameters
def build_network(arch, gpu, hidden_units, learning_rate):
    if arch == 'densenet121':
        model=models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model=models.vgg16(pretrained=True)
    else:
        print('Select either vgg16 or densenet121')

    # Freezing parameters 
    for param in model.parameters(): 
        param.requires_grad = False

    # Defining architecture for neural network model
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(1024, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('d1', nn.Dropout(p=0.2)),
                        ('fc2', nn.Linear(hidden_units, 500)),
                        ('relu', nn.ReLU()),
                        ('d2', nn.Dropout(p=0.2)),
                        ('fc3', nn.Linear(500, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))

    # Assigning classifier parameters to model
    model.classifier = classifier
    # Utilize GPU if available for acclerated compute speeds
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Defining criterion and optimizer 
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    model.to(gpu)
    return criterion, optimizer, model


def train_network(criterion, optimizer, epochs, model, trainloader, validation_loader): 
    # Training network
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 50
    for e in range (epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            model.train()
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad () # Optimizer is working on classifier parameters only
    
            # Forward pass and backward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # Track the loss and accuracy on the validation set to determine optimal hyperparameters
            if steps % print_every == 0:
                model.eval() # Switching to evaluation mode so that dropout is turned off
            
                # Turning off gradients for validation to save memory and computations
                with torch.no_grad():
                
                    model.to('cuda')
                
                    valid_loss = 0
                    accuracy = 0
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        output = model.forward(inputs)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Valid Loss: {:.3f}.. ".format(valid_loss/len(validation_loader)),
                    "Valid Accuracy: {:.3f}%".format(accuracy/len(validation_loader)*100))
            
                running_loss = 0
            
                model.train()
                
#

def measure_accuracy(testing_loader, model):
    correct_prediction, totals = 0 , 0
    model.to('cuda')
    with torch.no_grad():
        model.eval()
        for data in testing_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            totals += labels.size(0)
            correct_prediction += (prediction == labels).sum().item()
    print("Accuracy of neural network on testing dataset: %d %%" % (100 * correct_prediction / totals))
    return model  

#image_datasets = trainloader, testing_loader, validation_loader, training_dataset

def saving(model, hidden_units, epochs, optimizer, learning_rate, training_dataset):
    # Saving the checkpoint 
    #model.device('cpu')
    model.class_to_idx = training_dataset.class_to_idx
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'hidden_units': hidden_units,
                  'index': model.class_to_idx,
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')
    print("Saving model...")
    
    
def main():


    # Command line interface arguments for user interaction and model modifications
    parser = argparse.ArgumentParser(description='Training Neural Network')
    parser.add_argument('data_dir', type=str, action='store', help = 'Source of image data')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', action='store', help='Set location for save point') 
    parser.add_argument('--arch', type=str, default='densenet121', help='Selection of model architecture')
    parser.add_argument('--learning_rate', type = float, default=0.001, help = 'Set value for Learning Rate, default is 0.001')
    parser.add_argument('--hidden_units', type = int, default=1024, help = 'Set value for hidden unit amount, default is 1024')
    parser.add_argument('--epochs', type = int, default=10, help = 'Set value for EPOCHS, default is 10')
    parser.add_argument('--gpu', type = str, default='cuda', help = 'Choose device for location of processing.') 
    arg = parser.parse_args()
    data_dir = arg.data_dir
    save_dir = arg.save_dir
    arch = arg.arch
    learning_rate = arg.learning_rate
    hidden_units = arg.hidden_units
    epochs = arg.epochs
    gpu = arg.gpu
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        class_to_idx = cat_to_name
    
    trainloader, testing_loader, validation_loader, training_dataset = gather_data(arg.data_dir)
    criterion, optimizer, model = build_network(arch, gpu, hidden_units, learning_rate)
    train_network(criterion, optimizer, epochs, model, trainloader, validation_loader)
    model = measure_accuracy(testing_loader, model)
    saving(model, hidden_units, epochs, optimizer, learning_rate, training_dataset)
    print('Training sequence complete')
    
# Calling main function to run program
if __name__ == '__main__':
    main()
