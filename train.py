
import numpy as np
import torch
from torch import nn
import argparse
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'CNN model architecture to use')
#     parser.add_argument('--device', type = str, default = 'gpu', help = 'gpu or cpu')
    parser.add_argument('--lr', type = int, default = 0.002, help = 'learning rate')
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs')
    
    return parser.parse_args()


def main(data_directory):
    
    in_arg = get_input_args()
    
    # define transforms to be done on training set
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    # Setting up our training data and train loader
    
    train_data = datasets.ImageFolder(data_directory, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Building the classifier
    if in_arg.arch == "densenet":
        model = models.densenet161(pretrained=True)
    elif in_arg.arch == "vgg":
        model = models.vgg(pretrained=True)
    else:
        print("We can only handle vgg and densenet")

# Here we freeze parameters so we dont backprogatate on them
    for param in model.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2208, 500)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(500, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.lr)

    model.to(device);


    for epoch in range(in_arg(epochs)):
        running_loss = 0

        for inputs, labels in trainloader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            # Turn off gradients for validation
            with torch.no_grad():
                model.eval()
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    model.train()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/len(trainloader):.3f}.. "
                          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader):.3f}")

if __name__ == '__main__':
    arg1 = sys.argv[1]
    main(arg1)
    
