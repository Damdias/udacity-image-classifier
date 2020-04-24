import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from PIL import Image
import json
from networkUtil import parseCli,getimageDataset,loadModel,validation


args = parseCli()  
print("argumenet parser is complete")
device = args.gpu

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

image_datasets = getimageDataset(args)
print("transform images is complete")

trainloader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size =32,shuffle = True)
testloader = torch.utils.data.DataLoader(image_datasets["test"], batch_size = 20, shuffle = True)    
print("reading category json  is complete")   
    
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    

def main():
    
    model, arch = loadModel(args.arch, args.hidden_units)
    print("loading model is complete")
    criterion = nn.NLLLoss ()
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.learning_rate)

    if torch.cuda.is_available() and device:
        model.to ('cuda') 

    epochs = args.epochs
    print_every = 20
    steps = 0


    for e in range (epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate (trainloader):
            steps += 1
            if torch.cuda.is_available() and device:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad () 

            outputs = model.forward (inputs) 
            loss = criterion (outputs, labels) 
            loss.backward ()
            optimizer.step () 
            running_loss += loss.item () 

            if steps % print_every == 0:
                model.eval () 
                with torch.no_grad():
                    vlost, accuracy = validation(model, validationloader, criterion, device)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Valid Loss {:.4f}".format(vlost),
                   "valid Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
                model.train()

    model.to ('cpu') 
    model.class_to_idx = image_datasets["train"].class_to_idx
    
    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }

    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')


    

if __name__== "__main__":
    main()