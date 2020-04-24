import argparse
from torchvision import datasets, models, transforms
import torch
import numpy as np
from torch import nn
from torch import optim
from collections import OrderedDict

def parseCli():
    parser = argparse.ArgumentParser (description = "training script description")
    parser.add_argument ('data_dir', help = 'Name of the data directory path. Mandatory', type = str, default="flowers")
    parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used, OPtional', type = str,                     default="Alexnet")
    parser.add_argument ('--save_dir', help = 'Name of the saving directory path. Optional', type = str,default="store")
    parser.add_argument ('--learning_rate', help = 'Learning rate, default value 0.001', type = float, default=0.001)
    parser.add_argument ('--hidden_units', help = 'Number of Hidden units', type = int, default=4096)
    parser.add_argument ('--gpu', help = "Utilize GPU cuda, Optional", type = bool, default=True)
    parser.add_argument ('--category_names', help = "Category file name, Optional", type = str, default="cat_to_name.json")
    parser.add_argument ('--epochs', help = 'Number of epochs', type = int, default=10)
    return parser.parse_args ()

def prdictCli():
    parser = argparse.ArgumentParser (description = "predict script description")
    parser.add_argument ('image_path', help = 'Predict image path. Mandatory', type = str)
    parser.add_argument ('--category_names', help = "Category file name, Optional", type = str, default="cat_to_name.json")
    parser.add_argument ('--top_k', help = 'Number of top K classes, Optional', type = int, default=5)
    parser.add_argument ('--gpu', help = "Utilize GPU cuda, Optional", type = bool, default=True)
    parser.add_argument ('--save_dir', help = 'Name of the saving directory path. Optional', type = str,default="store")

    return parser.parse_args ()


def getimageDataset(args):
    data_dir = args.data_dir
    de_mean = [0.485, 0.456, 0.406]
    de_std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean= de_mean, std=de_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=de_mean, std=de_std)
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=de_mean, std=de_std)
         ])
    }
    image_datasets = {
        x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
    return image_datasets

def loadModel (arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential  (OrderedDict ([
                            ('con1', nn.Linear (25088, hidden_units)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('con2', nn.Linear (hidden_units, 2048)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('con3', nn.Linear (2048, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
  
    else: 
        arch = 'alexnet' 
        model = models.alexnet (pretrained = True)
        for parm in model.parameters():
            parm.requires_grad = False
        classifier = nn.Sequential  (OrderedDict ([
                                    ('con1', nn.Linear (9216, hidden_units)),
                                    ('relu1', nn.ReLU ()),
                                    ('dropout1', nn.Dropout (p = 0.3)),
                                    ('con2', nn.Linear (hidden_units, 2048)),
                                    ('relu2', nn.ReLU ()),
                                    ('dropout2', nn.Dropout (p = 0.3)),
                                    ('con3', nn.Linear (2048, 102)),
                                    ('output', nn.LogSoftmax (dim =1))
                                    ]))
    model.classifier = classifier
    return model, arch

def validation(model, valid_loader, criterion, device):
    if torch.cuda.is_available() and device:
       model.to('cuda')

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        if torch.cuda.is_available() and device:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy